using Genie.Router
using AutumnModelsController

route("/", AutumnModelsController.autumnmodels)

route("/playground", AutumnModelsController.playground, method = GET, named=:playground)

route("/compileautumn", AutumnModelsController.compileautumn, method = POST, named=:compile_autumn)

route("/startautumn", AutumnModelsController.startautumn, method = GET, named=:startautumn)

route("/clicked", AutumnModelsController.clicked, method = GET, named=:clicked)

route("/left", AutumnModelsController.left, method = GET, named=:left)

route("/right", AutumnModelsController.right, method = GET, named=:right)

route("/up", AutumnModelsController.up, method = GET, named=:up)

route("/down", AutumnModelsController.down, method = GET, named=:down)

route("/step", AutumnModelsController.step, method = GET, named=:step)

route("/replay", AutumnModelsController.replay, method = GET, named=:replay)

route("/random", AutumnModelsController.random, method = GET, named=:random)

route("/random2", AutumnModelsController.random2, method = GET, named=:random2)

route("/random3", AutumnModelsController.random3, method = GET, named=:random3)

route("/synthesize", AutumnModelsController.synthesize, method = GET, named=:synthesize)