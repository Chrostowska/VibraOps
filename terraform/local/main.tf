# Terraform configuration for local Docker deployment
terraform {
  required_version = ">= 1.0"
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
    local = {
      source  = "hashicorp/local"
      version = "~> 2.1"
    }
  }
}

# Configure the Docker Provider
provider "docker" {
  host = "unix:///var/run/docker.sock"
}

# Local variables
locals {
  app_name = "vibraops"
  environment = "local"
  
  tags = {
    Environment = local.environment
    Project     = local.app_name
    ManagedBy   = "terraform"
  }
}

# Create custom network
resource "docker_network" "vibraops_network" {
  name = "${local.app_name}-network"
  driver = "bridge"
  
  ipam_config {
    subnet = "172.20.0.0/16"
  }
}

# Redis container for caching
resource "docker_image" "redis" {
  name = "redis:alpine"
}

resource "docker_container" "redis" {
  name  = "${local.app_name}-redis"
  image = docker_image.redis.image_id
  
  ports {
    internal = 6379
    external = 6379
  }
  
  networks_advanced {
    name = docker_network.vibraops_network.name
  }
  
  restart = "unless-stopped"
  
  volumes {
    container_path = "/data"
    volume_name    = docker_volume.redis_data.name
  }
}

# Prometheus container
resource "docker_image" "prometheus" {
  name = "prom/prometheus:latest"
}

resource "docker_container" "prometheus" {
  name  = "${local.app_name}-prometheus"
  image = docker_image.prometheus.image_id
  
  ports {
    internal = 9090
    external = 9090
  }
  
  networks_advanced {
    name = docker_network.vibraops_network.name
  }
  
  restart = "unless-stopped"
  
  volumes {
    host_path      = abspath("../../monitoring/prometheus")
    container_path = "/etc/prometheus"
    read_only      = true
  }
  
  volumes {
    container_path = "/prometheus"
    volume_name    = docker_volume.prometheus_data.name
  }
  
  command = [
    "--config.file=/etc/prometheus/prometheus.yml",
    "--storage.tsdb.path=/prometheus",
    "--web.console.libraries=/etc/prometheus/console_libraries",
    "--web.console.templates=/etc/prometheus/consoles",
    "--storage.tsdb.retention.time=200h",
    "--web.enable-lifecycle"
  ]
}

# Grafana container
resource "docker_image" "grafana" {
  name = "grafana/grafana:latest"
}

resource "docker_container" "grafana" {
  name  = "${local.app_name}-grafana"
  image = docker_image.grafana.image_id
  
  ports {
    internal = 3000
    external = 3000
  }
  
  networks_advanced {
    name = docker_network.vibraops_network.name
  }
  
  restart = "unless-stopped"
  
  env = [
    "GF_SECURITY_ADMIN_PASSWORD=admin123",
    "GF_INSTALL_PLUGINS=grafana-clock-panel,grafana-simple-json-datasource",
    "GF_USERS_ALLOW_SIGN_UP=false",
    "GF_SERVER_DOMAIN=localhost",
    "GF_SMTP_ENABLED=false"
  ]
  
  volumes {
    container_path = "/var/lib/grafana"
    volume_name    = docker_volume.grafana_data.name
  }
  
  volumes {
    host_path      = abspath("../../monitoring/grafana/provisioning")
    container_path = "/etc/grafana/provisioning"
    read_only      = true
  }
  
  volumes {
    host_path      = abspath("../../monitoring/grafana/dashboards")
    container_path = "/var/lib/grafana/dashboards"
    read_only      = true
  }
  
  depends_on = [docker_container.prometheus]
}

# Loki container
resource "docker_image" "loki" {
  name = "grafana/loki:latest"
}

resource "docker_container" "loki" {
  name  = "${local.app_name}-loki"
  image = docker_image.loki.image_id
  
  ports {
    internal = 3100
    external = 3100
  }
  
  networks_advanced {
    name = docker_network.vibraops_network.name
  }
  
  restart = "unless-stopped"
  
  volumes {
    host_path      = abspath("../../monitoring/loki")
    container_path = "/etc/loki"
    read_only      = true
  }
  
  volumes {
    container_path = "/loki"
    volume_name    = docker_volume.loki_data.name
  }
  
  command = ["-config.file=/etc/loki/local-config.yaml"]
}

# Promtail container
resource "docker_image" "promtail" {
  name = "grafana/promtail:latest"
}

resource "docker_container" "promtail" {
  name  = "${local.app_name}-promtail"
  image = docker_image.promtail.image_id
  
  networks_advanced {
    name = docker_network.vibraops_network.name
  }
  
  restart = "unless-stopped"
  
  volumes {
    host_path      = abspath("../../logs")
    container_path = "/var/log/vibraops"
    read_only      = true
  }
  
  volumes {
    host_path      = abspath("../../monitoring/promtail")
    container_path = "/etc/promtail"
    read_only      = true
  }
  
  command = ["-config.file=/etc/promtail/config.yml"]
  
  depends_on = [docker_container.loki]
}

# Docker volumes
resource "docker_volume" "redis_data" {
  name = "${local.app_name}-redis-data"
}

resource "docker_volume" "prometheus_data" {
  name = "${local.app_name}-prometheus-data"
}

resource "docker_volume" "grafana_data" {
  name = "${local.app_name}-grafana-data"
}

resource "docker_volume" "loki_data" {
  name = "${local.app_name}-loki-data"
}

# Local file provisioning
resource "local_file" "environment_config" {
  filename = "../../.env.local"
  content = templatefile("${path.module}/templates/env.tpl", {
    redis_host = "localhost"
    redis_port = 6379
    prometheus_url = "http://localhost:9090"
    grafana_url = "http://localhost:3000"
    loki_url = "http://localhost:3100"
    api_url = "http://localhost:8000"
  })
}

# Output important information
output "services" {
  value = {
    api = {
      url = "http://localhost:8000"
      docs = "http://localhost:8000/docs"
      health = "http://localhost:8000/health"
    }
    grafana = {
      url = "http://localhost:3000"
      username = "admin"
      password = "admin123"
    }
    prometheus = {
      url = "http://localhost:9090"
    }
    redis = {
      host = "localhost"
      port = 6379
    }
  }
  description = "Service endpoints and access information"
} 