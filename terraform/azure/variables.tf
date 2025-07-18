# Input variables for Azure deployment

variable "environment" {
  description = "Environment name (e.g., dev, staging, prod)"
  type        = string
  default     = "dev"
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "azure_region" {
  description = "Azure region for resource deployment"
  type        = string
  default     = "East US"
}

variable "api_container_image" {
  description = "Container image for the VibraOps API"
  type        = string
  default     = "vibraops/api:latest"
}

variable "create_container_registry" {
  description = "Whether to create an Azure Container Registry"
  type        = bool
  default     = false
}

variable "redis_sku_name" {
  description = "Redis cache SKU name"
  type        = string
  default     = "Basic"
  
  validation {
    condition     = contains(["Basic", "Standard", "Premium"], var.redis_sku_name)
    error_message = "Redis SKU must be one of: Basic, Standard, Premium."
  }
}

variable "redis_capacity" {
  description = "Redis cache capacity"
  type        = number
  default     = 0
  
  validation {
    condition     = var.redis_capacity >= 0 && var.redis_capacity <= 6
    error_message = "Redis capacity must be between 0 and 6."
  }
}

variable "log_retention_days" {
  description = "Log Analytics workspace retention in days"
  type        = number
  default     = 30
  
  validation {
    condition     = var.log_retention_days >= 30 && var.log_retention_days <= 730
    error_message = "Log retention must be between 30 and 730 days."
  }
}

variable "container_cpu" {
  description = "CPU allocation for API container"
  type        = string
  default     = "1.0"
}

variable "container_memory" {
  description = "Memory allocation for API container (in GB)"
  type        = string
  default     = "2.0"
}

variable "enable_diagnostics" {
  description = "Enable diagnostics and monitoring"
  type        = bool
  default     = true
}

variable "allowed_ip_ranges" {
  description = "List of IP ranges allowed to access the API"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "tags" {
  description = "Additional tags to apply to resources"
  type        = map(string)
  default     = {}
} 