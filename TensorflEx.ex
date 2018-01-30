defmodule TensorflEx do
  @on_load :load_nifs

  def load_nifs do
    :erlang.load_nif("priv/TensorflEx", 0)
  end

  def version do
    raise "NIF version/0 not implemented"
  end

end
