using Genie.Router
using AutumnModelsController

route("/", AutumnModelsController.autumnmodels)

route("/compileautumn", AutumnModelsController.compileautumn, method = POST, named=:compile_autumn)

route("/runautumn", AutumnModelsController.runautumn, method = POST, named=:run_autumn)

route("/stopautumn", AutumnModelsController.stopautumn, method = POST, named=:stop_autumn)

route("/clicked", AutumnModelsController.clicked, method = POST, named=:clicked)

route("/step", AutumnModelsController.step, method = POST, named=:step)