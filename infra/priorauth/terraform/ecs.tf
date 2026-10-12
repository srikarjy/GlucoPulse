locals {
  image = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"
  common_env = [
    { name = "PRIORAUTH_DB_HOST", value = aws_db_instance.main.address },
    { name = "PRIORAUTH_DB_NAME", value = "priorauth" },
    { name = "PRIORAUTH_DB_USER", value = "priorauth" },
    { name = "PRIORAUTH_BACKEND", value = var.backend },
    { name = "PRIORAUTH_MCP_URL", value = "http://mcp.priorauth.local:8765/mcp" },
    { name = "PRIORAUTH_ENV", value = var.environment },
    { name = "AWS_REGION", value = var.region },
  ]
  common_secrets = [
    { name = "PRIORAUTH_DB_PASSWORD", valueFrom = aws_secretsmanager_secret.db.arn },
    { name = "PRIORAUTH_JWT_SECRET", valueFrom = aws_secretsmanager_secret.jwt.arn },
  ]
  services = {
    api    = { port = 8080, cpu = 512, memory = 1024, command = ["uvicorn", "--factory", "priorauth.api:app_factory", "--host", "0.0.0.0", "--port", "8080"] }
    worker = { port = 9102, cpu = 1024, memory = 2048, command = ["python", "-m", "priorauth.worker"] }
    mcp    = { port = 8765, cpu = 512, memory = 1024, command = ["python", "-m", "priorauth.mcp_server.server"] }
  }
}

resource "aws_ecs_cluster" "main" {
  name = "priorauth"
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

resource "aws_cloudwatch_log_group" "svc" {
  for_each          = local.services
  name              = "/priorauth/${var.environment}/${each.key}"
  retention_in_days = 365
  kms_key_id        = aws_kms_key.main.arn
}

resource "aws_ecs_task_definition" "svc" {
  for_each                 = local.services
  family                   = "priorauth-${each.key}"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = each.value.cpu
  memory                   = each.value.memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task[each.key].arn
  runtime_platform {
    cpu_architecture        = "X86_64"
    operating_system_family = "LINUX"
  }
  container_definitions = jsonencode([{
    name                   = each.key
    image                  = local.image
    essential              = true
    command                = each.value.command
    readonlyRootFilesystem = true
    user                   = "10001"
    portMappings           = [{ containerPort = each.value.port, protocol = "tcp" }]
    environment            = concat(local.common_env, [{ name = "PORT", value = tostring(each.value.port) }])
    secrets                = local.common_secrets
    linuxParameters        = { initProcessEnabled = true, capabilities = { drop = ["ALL"] } }
    stopTimeout            = 120 # worker drains in-flight pipelines on SIGTERM
    logConfiguration = {
      logDriver = "awslogs"
      options   = { awslogs-group = aws_cloudwatch_log_group.svc[each.key].name, awslogs-region = var.region, awslogs-stream-prefix = each.key }
    }
  }])
}

resource "aws_service_discovery_private_dns_namespace" "local" {
  name = "priorauth.local"
  vpc  = aws_vpc.main.id
}

resource "aws_service_discovery_service" "mcp" {
  name = "mcp"
  dns_config {
    namespace_id = aws_service_discovery_private_dns_namespace.local.id
    dns_records {
      type = "A"
      ttl  = 10
    }
  }
  health_check_custom_config { failure_threshold = 1 }
}

resource "aws_ecs_service" "svc" {
  for_each        = local.services
  name            = "priorauth-${each.key}"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.svc[each.key].arn
  launch_type     = "FARGATE"
  desired_count   = each.key == "api" ? var.api_desired_count : each.key == "worker" ? var.worker_min : 2

  deployment_minimum_healthy_percent = 100
  deployment_maximum_percent         = 200
  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }
  network_configuration {
    subnets         = aws_subnet.private[*].id
    security_groups = [each.key == "api" ? aws_security_group.api.id : each.key == "worker" ? aws_security_group.worker.id : aws_security_group.mcp.id]
  }
  dynamic "load_balancer" {
    for_each = each.key == "api" ? [1] : []
    content {
      target_group_arn = aws_lb_target_group.api.arn
      container_name   = "api"
      container_port   = 8080
    }
  }
  dynamic "service_registries" {
    for_each = each.key == "mcp" ? [1] : []
    content { registry_arn = aws_service_discovery_service.mcp.arn }
  }
  lifecycle { ignore_changes = [task_definition, desired_count] } # CI registers new revisions; autoscaling owns count
}

# ---- API behind an HTTPS-only ALB (MCP and worker are never publicly reachable) ----
resource "aws_lb" "api" {
  name                       = "priorauth-${var.environment}"
  load_balancer_type         = "application"
  internal                   = false
  subnets                    = aws_subnet.public[*].id
  security_groups            = [aws_security_group.alb.id]
  drop_invalid_header_fields = true
  enable_deletion_protection = true
}

resource "aws_lb_target_group" "api" {
  name        = "priorauth-api-${var.environment}"
  port        = 8080
  protocol    = "HTTP"
  target_type = "ip"
  vpc_id      = aws_vpc.main.id
  health_check {
    path    = "/readyz"
    matcher = "200"
  }
  deregistration_delay = 30
}

resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.api.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = var.certificate_arn
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}

# /metrics is for Prometheus inside the VPC, never the internet
resource "aws_lb_listener_rule" "block_metrics" {
  listener_arn = aws_lb_listener.https.arn
  priority     = 1
  action {
    type = "fixed-response"
    fixed_response {
      content_type = "text/plain"
      message_body = "not found"
      status_code  = "404"
    }
  }
  condition {
    path_pattern { values = ["/metrics"] }
  }
}

# ---- autoscaling ----
resource "aws_appautoscaling_target" "api" {
  service_namespace  = "ecs"
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.svc["api"].name}"
  scalable_dimension = "ecs:service:DesiredCount"
  min_capacity       = var.api_desired_count
  max_capacity       = var.api_desired_count * 4
}

resource "aws_appautoscaling_policy" "api_cpu" {
  name               = "api-cpu"
  policy_type        = "TargetTrackingScaling"
  service_namespace  = aws_appautoscaling_target.api.service_namespace
  resource_id        = aws_appautoscaling_target.api.resource_id
  scalable_dimension = aws_appautoscaling_target.api.scalable_dimension
  target_tracking_scaling_policy_configuration {
    target_value = 60
    predefined_metric_specification { predefined_metric_type = "ECSServiceAverageCPUUtilization" }
  }
}

resource "aws_appautoscaling_target" "worker" {
  service_namespace  = "ecs"
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.svc["worker"].name}"
  scalable_dimension = "ecs:service:DesiredCount"
  min_capacity       = var.worker_min
  max_capacity       = var.worker_max
}

# CPU is a weak proxy for backlog; scale on queue depth by publishing pa_queue_jobs{state="queued"} as a CloudWatch metric
# (ADOT collector sidecar or the EKS/KEDA path in infra/priorauth/k8s). CPU target tracking is the safe default here.
resource "aws_appautoscaling_policy" "worker_cpu" {
  name               = "worker-cpu"
  policy_type        = "TargetTrackingScaling"
  service_namespace  = aws_appautoscaling_target.worker.service_namespace
  resource_id        = aws_appautoscaling_target.worker.resource_id
  scalable_dimension = aws_appautoscaling_target.worker.scalable_dimension
  target_tracking_scaling_policy_configuration {
    target_value = 55
    predefined_metric_specification { predefined_metric_type = "ECSServiceAverageCPUUtilization" }
  }
}

# ---- alarms (CloudWatch; Prometheus alerts cover app-level signals) ----
resource "aws_cloudwatch_metric_alarm" "api_5xx" {
  alarm_name          = "priorauth-${var.environment}-api-5xx"
  namespace           = "AWS/ApplicationELB"
  metric_name         = "HTTPCode_Target_5XX_Count"
  dimensions          = { LoadBalancer = aws_lb.api.arn_suffix }
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 2
  threshold           = 5
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
}

resource "aws_cloudwatch_metric_alarm" "db_cpu" {
  alarm_name          = "priorauth-${var.environment}-db-cpu"
  namespace           = "AWS/RDS"
  metric_name         = "CPUUtilization"
  dimensions          = { DBInstanceIdentifier = aws_db_instance.main.identifier }
  statistic           = "Average"
  period              = 300
  evaluation_periods  = 3
  threshold           = 80
  comparison_operator = "GreaterThanThreshold"
}
