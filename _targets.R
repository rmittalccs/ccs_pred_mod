# _targets.R file
library(targets)

source("./functions.R")

list(
  tar_target(
    name = second_target,
    command = outer_function(first_target) + 2
  ),
  tar_target(
    name = first_target,
    command = 2
  ),
  tar_target(
    name = simple,
    command = my_name("Rupal")
  ),
  tar_target(
    name = python_execution,
    command = tryCatch(python_script(python_executable), 
                       error = function(e) message("Error executing Python script:", e))
  )
)
