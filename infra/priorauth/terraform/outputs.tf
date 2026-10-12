output "alb_dns_name" { value = aws_lb.api.dns_name }
output "ecr_repository_url" { value = aws_ecr_repository.app.repository_url }
output "cluster_name" { value = aws_ecs_cluster.main.name }
output "db_endpoint" { value = aws_db_instance.main.address }
