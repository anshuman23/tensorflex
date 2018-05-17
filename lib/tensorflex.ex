defmodule Tensorflex do
  @on_load :load_nifs

  def load_nifs do
    :erlang.load_nif("priv/Tensorflex", 0)
  end

  def version do
    raise "NIF tf_version/0 not implemented"
  end

  def new_graph do
    raise "NIF tf_new_graph/0 not implemented"
  end

  def new_op(_graph, _op, _label) do
    raise "NIF tf_new_op/3 not implemented"
  end

  def read_graph(_filepath) do
    raise "NIF read_graph/1 not implemented"
  end

  def string_constant(_value) do
    raise "NIF tf_string_constant/1 not implemented"
  end

  def create_and_run_sess(_graph, _opdesc, _tensor) do
    raise "NIF tf_create_and_run_sess/3 not implemented"
  end

end
