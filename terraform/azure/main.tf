# Terraform configuration for Azure deployment
terraform {
  required_version = ">= 1.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.1"
    }
  }
}

# Configure the Microsoft Azure Provider
provider "azurerm" {
  features {
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
  }
}

# Local variables
locals {
  app_name = "vibraops"
  environment = var.environment
  location = var.azure_region
  
  common_tags = {
    Environment = local.environment
    Project     = local.app_name
    ManagedBy   = "terraform"
    CreatedDate = formatdate("YYYY-MM-DD", timestamp())
  }
}

# Random suffix for unique naming
resource "random_string" "suffix" {
  length  = 6
  special = false
  upper   = false
}

# Resource Group
resource "azurerm_resource_group" "vibraops" {
  name     = "rg-${local.app_name}-${local.environment}-${random_string.suffix.result}"
  location = local.location
  tags     = local.common_tags
}

# Virtual Network
resource "azurerm_virtual_network" "vibraops" {
  name                = "vnet-${local.app_name}-${local.environment}"
  address_space       = ["10.0.0.0/16"]
  location            = azurerm_resource_group.vibraops.location
  resource_group_name = azurerm_resource_group.vibraops.name
  tags                = local.common_tags
}

# Subnet for Container Instances
resource "azurerm_subnet" "containers" {
  name                 = "subnet-containers"
  resource_group_name  = azurerm_resource_group.vibraops.name
  virtual_network_name = azurerm_virtual_network.vibraops.name
  address_prefixes     = ["10.0.1.0/24"]
  
  delegation {
    name = "Microsoft.ContainerInstance/containerGroups"
    service_delegation {
      name    = "Microsoft.ContainerInstance/containerGroups"
      actions = ["Microsoft.Network/virtualNetworks/subnets/action"]
    }
  }
}

# Application Insights for monitoring
resource "azurerm_application_insights" "vibraops" {
  name                = "ai-${local.app_name}-${local.environment}"
  location            = azurerm_resource_group.vibraops.location
  resource_group_name = azurerm_resource_group.vibraops.name
  application_type    = "web"
  tags                = local.common_tags
}

# Log Analytics Workspace
resource "azurerm_log_analytics_workspace" "vibraops" {
  name                = "law-${local.app_name}-${local.environment}"
  location            = azurerm_resource_group.vibraops.location
  resource_group_name = azurerm_resource_group.vibraops.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
  tags                = local.common_tags
}

# Redis Cache for session storage and caching
resource "azurerm_redis_cache" "vibraops" {
  name                = "redis-${local.app_name}-${local.environment}-${random_string.suffix.result}"
  location            = azurerm_resource_group.vibraops.location
  resource_group_name = azurerm_resource_group.vibraops.name
  capacity            = 0
  family              = "C"
  sku_name            = "Basic"
  enable_non_ssl_port = false
  minimum_tls_version = "1.2"
  tags                = local.common_tags
}

# Storage Account for model artifacts and logs
resource "azurerm_storage_account" "vibraops" {
  name                     = "st${local.app_name}${local.environment}${random_string.suffix.result}"
  resource_group_name      = azurerm_resource_group.vibraops.name
  location                 = azurerm_resource_group.vibraops.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  
  blob_properties {
    cors_rule {
      allowed_headers    = ["*"]
      allowed_methods    = ["DELETE", "GET", "HEAD", "MERGE", "POST", "OPTIONS", "PUT"]
      allowed_origins    = ["*"]
      exposed_headers    = ["*"]
      max_age_in_seconds = 200
    }
  }
  
  tags = local.common_tags
}

# Container for model storage
resource "azurerm_storage_container" "models" {
  name                  = "models"
  storage_account_name  = azurerm_storage_account.vibraops.name
  container_access_type = "private"
}

# Container for logs
resource "azurerm_storage_container" "logs" {
  name                  = "logs"
  storage_account_name  = azurerm_storage_account.vibraops.name
  container_access_type = "private"
}

# Container Registry (optional, for custom images)
resource "azurerm_container_registry" "vibraops" {
  count               = var.create_container_registry ? 1 : 0
  name                = "acr${local.app_name}${local.environment}${random_string.suffix.result}"
  resource_group_name = azurerm_resource_group.vibraops.name
  location            = azurerm_resource_group.vibraops.location
  sku                 = "Basic"
  admin_enabled       = true
  tags                = local.common_tags
}

# Container Group for VibraOps API
resource "azurerm_container_group" "vibraops_api" {
  name                = "ci-${local.app_name}-api-${local.environment}"
  location            = azurerm_resource_group.vibraops.location
  resource_group_name = azurerm_resource_group.vibraops.name
  ip_address_type     = "Public"
  dns_name_label      = "${local.app_name}-api-${local.environment}-${random_string.suffix.result}"
  os_type             = "Linux"
  
  subnet_ids = [azurerm_subnet.containers.id]
  
  container {
    name   = "vibraops-api"
    image  = var.api_container_image
    cpu    = "1.0"
    memory = "2.0"
    
    ports {
      port     = 8000
      protocol = "TCP"
    }
    
    environment_variables = {
      PYTHONPATH         = "/app/src"
      MODEL_PATH         = "/app/models"
      LOG_LEVEL          = "INFO"
      REDIS_HOST         = azurerm_redis_cache.vibraops.hostname
      REDIS_PORT         = azurerm_redis_cache.vibraops.port
      ENVIRONMENT        = local.environment
    }
    
    secure_environment_variables = {
      REDIS_PASSWORD = azurerm_redis_cache.vibraops.primary_access_key
      APPINSIGHTS_INSTRUMENTATIONKEY = azurerm_application_insights.vibraops.instrumentation_key
    }
    
    volume {
      name       = "models"
      mount_path = "/app/models"
      read_only  = true
      
      storage_account_name = azurerm_storage_account.vibraops.name
      storage_account_key  = azurerm_storage_account.vibraops.primary_access_key
      share_name          = azurerm_storage_share.models.name
    }
    
    volume {
      name       = "logs"
      mount_path = "/app/logs"
      read_only  = false
      
      storage_account_name = azurerm_storage_account.vibraops.name
      storage_account_key  = azurerm_storage_account.vibraops.primary_access_key
      share_name          = azurerm_storage_share.logs.name
    }
  }
  
  diagnostics {
    log_analytics {
      workspace_id  = azurerm_log_analytics_workspace.vibraops.workspace_id
      workspace_key = azurerm_log_analytics_workspace.vibraops.primary_shared_key
    }
  }
  
  tags = local.common_tags
}

# File shares for persistent storage
resource "azurerm_storage_share" "models" {
  name                 = "models"
  storage_account_name = azurerm_storage_account.vibraops.name
  quota                = 50
}

resource "azurerm_storage_share" "logs" {
  name                 = "logs"
  storage_account_name = azurerm_storage_account.vibraops.name
  quota                = 50
}

# Network Security Group
resource "azurerm_network_security_group" "vibraops" {
  name                = "nsg-${local.app_name}-${local.environment}"
  location            = azurerm_resource_group.vibraops.location
  resource_group_name = azurerm_resource_group.vibraops.name
  
  security_rule {
    name                       = "HTTP"
    priority                   = 1001
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "8000"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }
  
  security_rule {
    name                       = "HTTPS"
    priority                   = 1002
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "443"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }
  
  tags = local.common_tags
}

# Associate NSG with subnet
resource "azurerm_subnet_network_security_group_association" "containers" {
  subnet_id                 = azurerm_subnet.containers.id
  network_security_group_id = azurerm_network_security_group.vibraops.id
} 