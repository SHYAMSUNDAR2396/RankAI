terraform {
  required_version = ">= 1.6.0"
  required_providers {
    alicloud = {
      source  = "aliyun/alicloud"
      version = "~> 1.200"
    }
  }
}

provider "alicloud" {
  region = var.region
}

variable "region" {
  description = "Alibaba Cloud region (Shanghai or Singapore recommended for Qwen Cloud proximity)."
  type        = string
  default     = "cn-shanghai"
}

variable "instance_type" {
  description = "ECS instance type — 2 vCPU / 4 GiB is sufficient for the Qwen Cloud + ChromaDB stack."
  type        = string
  default     = "ecs.t6-c1m2.large"
}

variable "image_id" {
  description = "ECS image ID — Alibaba Cloud Linux 3 (3.2104) recommended."
  type        = string
  default     = "aliyun_3_x64_20G_alibase_20251130.vhd"
}

variable "key_pair_name" {
  description = "ECS SSH key pair name; create with `aliyun ecs CreateKeyPair` if absent."
  type        = string
}

resource "alicloud_vpc" "rankai" {
  vpc_name   = "rankai-vpc"
  cidr_block = "10.20.0.0/16"
}

resource "alicloud_vswitch" "rankai" {
  vpc_id            = alicloud_vpc.rankai.id
  cidr_block        = "10.20.1.0/24"
  zone_id           = "${var.region}a"
  vswitch_name      = "rankai-vswitch"
}

resource "alicloud_security_group" "rankai" {
  name        = "rankai-sg"
  vpc_id      = alicloud_vpc.rankai.id
  description = "RankAI demo security group"

  ingress {
    protocol    = "tcp"
    port_range  = "8080/8080"
    cidr_blocks = ["0.0.0.0/0"]
    description = "RankAI dashboard"
  }

  ingress {
    protocol    = "tcp"
    port_range  = "22/22"
    cidr_blocks = ["0.0.0.0/0"]
    description = "SSH"
  }

  egress {
    protocol    = "all"
    port_range  = "-1/-1"
    cidr_blocks = ["0.0.0.0/0"]
,  description = "Outbound to Qwen Cloud + OSS"
  }
}

resource "alicloud_ecs_command" "rankai_bootstrap" {
  type             = "RunShellScript"
  command_content  = <<-EOT
    set -euo pipefail
    dnf install -y docker docker-compose-plugin git
    systemctl enable --now docker
    git clone https://github.com/<OWNER>/RankAI.git /opt/rankai || true
    cd /opt/rankai
    echo 'DASHSCOPE_API_KEY=${var.dashscope_api_key}' > .env
    echo 'LLM_BACKEND=qwen_cloud' >> .env
    docker compose -f deploy/alicloud/docker-compose.yml --env-file .env up -d
    curl -fsS http://localhost:8080/healthz
  EOT
  timeout          = 600
  enable           = true
  description      = "Bootstrap RankAI via docker compose on ECS"
  lifecycle {
    ignore_changes = [command_content]
  }
}

resource "alicloud_ecs_instance" "rankai" {
  image_id                   = var.image_id
  instance_type              = var.instance_type
  vswitch_id                 = alicloud_vswitch.rankai.id
  security_groups            = [alicloud_security_group.rankai.id]
  key_name                   = var.key_pair_name
  instance_name              = "rankai-demo"
  internet_max_bandwidth_out = 10
  password                   = ""
  internet_charge_type       = "PayByTraffic"
  tags = {
    Project     = "RankAI"
    Environment = "hackathon"
    Track       = "4-autopilot-agent"
  }
}

resource "alicloud_eip" "rankai" {
  bandwidth            = 10
  internet_charge_type = "PayByTraffic"
  instance_id          = alicloud_ecs_instance.rankai.id
}

output "rankai_console_url" {
  value       = "http://${alicloud_eip.rankai.ip_address}:8080"
  description = "Public URL of the deployed RankAI demo."
}

output "rankai_instance_id" {
  value       = alicloud_ecs_instance.rankai.id
  description = "Underlying ECS instance ID for live demo verification."
}

output "rankai_region" {
  value       = var.region
  description = "Alibaba Cloud region the deployment was provisioned in."
}
