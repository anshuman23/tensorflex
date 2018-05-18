defmodule Tensorflex do
  @on_load :load_nifs

  def load_nifs do
    :erlang.load_nif("priv/Tensorflex", 0)
  end

  def version do
    raise "NIF tf_version/0 not implemented"
  end

  def read_graph(_filepath) do
    raise "NIF read_graph/1 not implemented"
  end

  def get_graph_ops(_graph) do
    raise "NIF get_graph_ops/1 not implemented"
  end
  
end
