  cd(@__DIR__)
  import Pkg
  Pkg.activate(".")

  function main()
    include(joinpath("src", "CausalDiscoveryApp.jl"))
  end; main()
