import Genie
import Logging
import Dates

function initialize_logging()
  date_format = "yyyy-mm-dd HH:MM:SS"

  logger =  if Genie.config.log_to_file
              isdir(Genie.config.path_log) || mkpath(Genie.config.path_log)

            else
              Logging.ConsoleLogger(stdout, Genie.config.log_level)
            end

  nothing
end

initialize_logging()