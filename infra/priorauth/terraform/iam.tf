data "aws_iam_policy_document" "ecs_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

# Execution role: pull image, write logs, read ONLY the two secrets at container start
resource "aws_iam_role" "execution" {
  name               = "priorauth-${var.environment}-exec"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

resource "aws_iam_role_policy_attachment" "execution" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "execution_secrets" {
  role = aws_iam_role.execution.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Effect = "Allow", Action = ["secretsmanager:GetSecretValue"], Resource = [aws_secretsmanager_secret.db.arn, aws_secretsmanager_secret.jwt.arn] },
      { Effect = "Allow", Action = ["kms:Decrypt"], Resource = aws_kms_key.main.arn },
    ]
  })
}

# Task roles: separate per service. Only worker + mcp-adjacent roles may call Bedrock, and only the listed models.
resource "aws_iam_role" "task" {
  for_each           = toset(["api", "worker", "mcp"])
  name               = "priorauth-${var.environment}-${each.key}"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

resource "aws_iam_role_policy" "worker_bedrock" {
  count = length(var.bedrock_model_arns) > 0 ? 1 : 0
  role  = aws_iam_role.task["worker"].id
  policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Action = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"], Resource = var.bedrock_model_arns }]
  })
}

resource "aws_iam_role_policy" "task_telemetry" {
  for_each = aws_iam_role.task
  role     = each.value.id
  policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Action = ["cloudwatch:PutMetricData"], Resource = "*", Condition = { StringEquals = { "cloudwatch:namespace" = "PriorAuth" } } }]
  })
}
