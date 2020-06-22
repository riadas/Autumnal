using Genie.Router
using AutumnModelsController

route("/", AutumnModelsController.autumnmodels)

route("/compileautumn", AutumnModelsController.compileautumn, method = POST, named=:compile_autumn)