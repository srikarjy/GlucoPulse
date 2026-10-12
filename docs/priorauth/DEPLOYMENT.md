# Deployment

All three paths run the same image (`deploy/priorauth/Dockerfile`; entrypoints `api` | `worker` | `mcp`).

## Local: Docker Compose (verified)
`docker compose -f docker-compose.priorauth.yml up -d --build`, then `python -m priorauth.scripts.smoke`. Requires `PRIORAUTH_JWT_SECRET`. Set `PRIORAUTH_BACKEND=bedrock` (plus AWS creds) to use the Strands agents.

## AWS ECS Fargate: `infra/priorauth/terraform` (validated, never applied)
VPC (2 AZ, private app/db subnets, NAT per AZ, flow logs), ALB (HTTPS/TLS1.3, CIDR allow-list, `/metrics` blocked), ECS services `api`/`worker`/`mcp` (MCP reachable only from workers via Cloud Map),
RDS PostgreSQL 16 Multi-AZ (encrypted, `force_ssl`, deletion protection; pgvector via `CREATE EXTENSION`), Secrets Manager, KMS CMK, ECR (immutable tags, scan-on-push), per-service task roles
(only `worker` may call the listed Bedrock models), circuit-breaker rollbacks, CPU autoscaling, CloudWatch alarms.

```bash
cd infra/priorauth/terraform
terraform init -backend-config="bucket=…" -backend-config="key=priorauth/staging.tfstate" -backend-config="region=us-east-1" -backend-config="dynamodb_table=…"
terraform plan -var certificate_arn=… -var 'alb_ingress_cidrs=["203.0.113.0/24"]'
```
Review the plan: this config has never been planned or applied. Before first apply, push an image tagged `bootstrap` (ECR tags are immutable) or services will not start.

## AWS EKS: `infra/priorauth/k8s` (validated, never applied)
`kubectl apply -k infra/priorauth/k8s` after replacing `ACCOUNT_ID`, the RDS host, certificate ARN and CIDRs. Needs: AWS Load Balancer Controller, External Secrets Operator, KEDA, kube-prometheus (or equivalent).
Pod Security `restricted`, default-deny NetworkPolicies, PDBs, IRSA for the worker. The worker scales on **queued-job count** via a KEDA Postgres trigger (the query was verified on real Postgres).

## CI/CD (`.github/workflows/priorauth-ci.yml`, linted but not yet run on GitHub)
`test` (ruff, pytest with a pgvector service, scenario-reproducibility check, **200-scenario eval gate**) → `image` (build, run as non-root and hit `/healthz`, Trivy) → `iac` (terraform fmt/validate, kubeconform) →
`deploy` (main only, gated by a GitHub Environment with reviewers; OIDC role, no long-lived keys; registers a new ECS task-definition revision per service with a SHA-tagged image and relies on the circuit breaker to roll back).

## Rollback
ECS: circuit breaker auto-rolls back a failing deployment; manual: `aws ecs update-service --task-definition priorauth-<svc>:<previous-revision>`. K8s: `kubectl rollout undo deploy/pa-<svc>`.
In-flight jobs survive restarts (leases expire and are reclaimed; tool and task writes are idempotent).
