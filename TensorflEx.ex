defmodule TensorflEx do
  @on_load :load_nifs

  def load_nifs do
    :erlang.load_nif("priv/TensorflEx", 0)
  end

  def tf_version do
    raise "NIF tf_version/0 not implemented"
  end

  def tf_new_graph do
    raise "NIF tf_new_graph/0 not implemented"
  end

  def tf_new_op(_graph, _op, _label) do
    raise "NIF tf_new_op/3 not implemented"
  end

  def tf_string_constant(_value) do
    raise "NIF tf_string_constant/1 not implemented"
  end

  def tf_create_and_run_sess(_graph, _opdesc, _tensor) do
    raise "NIF tf_create_and_run_sess/3 not implemented"
  end

end
