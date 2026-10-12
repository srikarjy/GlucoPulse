variable "region" {
  type    = string
  default = "us-east-1"
}
variable "environment" {
  type    = string
  default = "staging"
}
variable "image_tag" {
  description = "Initial image tag. CI registers new task-definition revisions afterwards (services ignore task_definition drift)."
  type        = string
  default     = "bootstrap"
}
variable "certificate_arn" {
  description = "ACM certificate for the ALB HTTPS listener"
  type        = string
}
variable "bedrock_model_arns" {
  description = "Foundation-model / inference-profile ARNs the agents may invoke. Empty = deterministic backend only."
  type        = list(string)
  default     = []
}
variable "backend" {
  description = "deterministic | bedrock"
  type        = string
  default     = "deterministic"
  validation {
    condition     = contains(["deterministic", "bedrock"], var.backend)
    error_message = "backend must be deterministic or bedrock."
  }
}
variable "vpc_cidr" {
  type    = string
  default = "10.42.0.0/16"
}
variable "db_instance_class" {
  type    = string
  default = "db.t4g.medium"
}
variable "api_desired_count" {
  type    = number
  default = 2
}
variable "worker_min" {
  type    = number
  default = 2
}
variable "worker_max" {
  type    = number
  default = 10
}
variable "alb_ingress_cidrs" {
  description = "CIDRs allowed to reach the API ALB (EHR / intake systems, reviewer UI). No default: must be explicit."
  type        = list(string)
}
