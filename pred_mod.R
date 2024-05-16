library(reticulate)

# Directly Running --------------------------------------------------------------------------------

# Set the path to your virtual environment
virtualenv_path <- "/home/RMittal@ccsfundraising.com/ccs_pred_mod/.py39_scikit"

Sys.setenv("RETICULATE_PYTHON" = "/home/RMittal@ccsfundraising.com/ccs_pred_mod/.py39_scikit/bin/python")
use_virtualenv(virtualenv_path, required=TRUE)
source_python("python/pred_mod.py", convert=TRUE)

# Set the Python executable path within the virtual environment
python_executable <- file.path(virtualenv_path, "bin", "python")

# Define the command to call the Python script with arguments
python_script <- paste(python_executable, "pred_mod.py -v 1", sep = " ")

# Execute the command
system(python_script)

# FYI --------------------------------------------------------------------------------
# library(reticulate)
# virtualenv_path <- "/path/to/myenv"
# virtualenv_install(virtualenv_path, "chardet")

