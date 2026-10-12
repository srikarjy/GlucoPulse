terraform {
  required_version = ">= 1.6"
  required_providers {
    aws    = { source = "hashicorp/aws", version = "~> 5.70" }
    random = { source = "hashicorp/random", version = "~> 3.6" }
  }
  # Configure remote state (S3 + DynamoDB lock) per environment:  terraform init -backend-config=...
  backend "s3" {}
}

provider "aws" {
  region = var.region
  default_tags {
    tags = { Project = "priorauth", Environment = var.environment, ManagedBy = "terraform", DataClass = "synthetic-only" }
  }
}
