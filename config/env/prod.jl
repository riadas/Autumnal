using Genie.Configuration, Logging

const config = Settings(
  server_port                     = 8000,
  server_host                     = "0.0.0.0",
  log_level                       = Logging.Error,
  log_to_file                     = true,
  server_handle_static_files      = false,
  cors_headers                    = Dict(
    "Access-Control-Allow-Origin" => "*",
    "Access-Control-Allow-Headers" => "Content-Type",
    "Access-Control-Allow-Methods" => "GET,POST,PUT,DELETE,OPTIONS")
)

ENV["JULIA_REVISE"] = "off"