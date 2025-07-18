# Output values for Azure deployment

output "resource_group_name" {
  description = "Name of the resource group"
  value       = azurerm_resource_group.vibraops.name
}

output "api_url" {
  description = "URL of the VibraOps API"
  value       = "http://${azurerm_container_group.vibraops_api.fqdn}:8000"
}

output "api_fqdn" {
  description = "Fully qualified domain name of the API"
  value       = azurerm_container_group.vibraops_api.fqdn
}

output "api_docs_url" {
  description = "URL of the API documentation"
  value       = "http://${azurerm_container_group.vibraops_api.fqdn}:8000/docs"
}

output "api_health_url" {
  description = "URL of the API health check"
  value       = "http://${azurerm_container_group.vibraops_api.fqdn}:8000/health"
}

output "redis_hostname" {
  description = "Redis cache hostname"
  value       = azurerm_redis_cache.vibraops.hostname
  sensitive   = true
}

output "redis_port" {
  description = "Redis cache port"
  value       = azurerm_redis_cache.vibraops.port
}

output "redis_primary_key" {
  description = "Redis cache primary access key"
  value       = azurerm_redis_cache.vibraops.primary_access_key
  sensitive   = true
}

output "storage_account_name" {
  description = "Name of the storage account"
  value       = azurerm_storage_account.vibraops.name
}

output "storage_account_primary_key" {
  description = "Primary access key for the storage account"
  value       = azurerm_storage_account.vibraops.primary_access_key
  sensitive   = true
}

output "application_insights_instrumentation_key" {
  description = "Application Insights instrumentation key"
  value       = azurerm_application_insights.vibraops.instrumentation_key
  sensitive   = true
}

output "application_insights_app_id" {
  description = "Application Insights application ID"
  value       = azurerm_application_insights.vibraops.app_id
}

output "log_analytics_workspace_id" {
  description = "Log Analytics workspace ID"
  value       = azurerm_log_analytics_workspace.vibraops.workspace_id
  sensitive   = true
}

output "container_registry_login_server" {
  description = "Container registry login server"
  value       = var.create_container_registry ? azurerm_container_registry.vibraops[0].login_server : null
}

output "container_registry_admin_username" {
  description = "Container registry admin username"
  value       = var.create_container_registry ? azurerm_container_registry.vibraops[0].admin_username : null
  sensitive   = true
}

output "container_registry_admin_password" {
  description = "Container registry admin password"
  value       = var.create_container_registry ? azurerm_container_registry.vibraops[0].admin_password : null
  sensitive   = true
}

output "virtual_network_id" {
  description = "Virtual network ID"
  value       = azurerm_virtual_network.vibraops.id
}

output "subnet_id" {
  description = "Container subnet ID"
  value       = azurerm_subnet.containers.id
}

output "deployment_info" {
  description = "Summary of deployment information"
  value = {
    environment           = local.environment
    location             = local.location
    resource_group       = azurerm_resource_group.vibraops.name
    api_endpoint         = "http://${azurerm_container_group.vibraops_api.fqdn}:8000"
    monitoring_enabled   = var.enable_diagnostics
    container_registry   = var.create_container_registry
  }
} 