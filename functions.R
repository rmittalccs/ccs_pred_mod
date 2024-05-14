
global_object <- 3

inner_function <- function(argument) {
  local_object <- 1
  argument + global_object + local_object + 2
}

outer_function <- function(object) {
  object + inner_function(object) + 1
}

my_name <- function(argument){
  paste("My name is", argument)
}

virtualenv_path <- "/home/RMittal@ccsfundraising.com/ccs_pred_mod/.py39_scikit"
python_executable <- file.path(virtualenv_path, "bin", "python")

#python_script <- paste(python_executable, "pred_mod.py -v 1", sep = " ")
python_script <- function(python_executable){
  run_python_script <- paste(python_executable, "pred_mod.py -v 1", sep = " ")
  #cat("Executing command:", run_python_script, "\n")
  system(run_python_script)
}
