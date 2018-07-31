defmodule Tensorflex.MixProject do
  use Mix.Project

  def project do
    [
      app: :tensorflex,
      version: "0.1.0",
      compilers: [:elixir_make] ++ Mix.compilers,
      elixir: "~> 1.6",
      build_embedded: Mix.env == :prod,
      start_permanent: Mix.env() == :prod,
      description: description(),
      package: package(),
      deps: deps()
    ]
  end

  defp description do
    """
    A simple and fast library for running Tensorflow graph models in Elixir. Tensorflex is written around the Tensorflow C API, and allows Elixir developers to leverage Machine Learning and Deep Learning solutions in their projects.
    """
  end
  
  defp package do
    [
      maintainers: ["Anshuman Chhabra"],
      files: ["lib", "priv", "mix.exs", "Makefile", "c_src", "README.md", "LICENSE"],
      licenses: ["Apache 2.0"],
      links: %{"GitHub" => "https://github.com/anshuman23/tensorflex"}
    ]
  end
  
  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      # {:dep_from_hexpm, "~> 0.3.0"},
      # {:dep_from_git, git: "https://github.com/elixir-lang/my_dep.git", tag: "0.1.0"},
      {:elixir_make, "~> 0.4", runtime: false},
      {:ex_doc, "~> 0.18.0", only: :dev, runtime: false}
    ]
  end
end
