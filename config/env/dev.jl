using Genie.Configuration, Logging

const config = Settings(
  server_port                     = 8000,
  server_host                     = "127.0.0.1",
  log_level                       = Logging.Debug,
  log_to_file                     = false,
  server_handle_static_files      = true,
  cors_headers                    = Dict(
    "Access-Control-Allow-Origin" => "*",
    "Access-Control-Allow-Headers" => "Content-Type",
    "Access-Control-Allow-Methods" => "GET,POST,PUT,DELETE,OPTIONS")
)

ENV["JULIA_REVISE"] = "auto"