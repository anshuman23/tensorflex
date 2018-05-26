defmodule Tensorflex do
  @on_load :load_nifs

  def load_nifs do
    :erlang.load_nif("priv/Tensorflex", 0)
  end

  def create_matrix(_nrows,_ncolumns, _list) do
    raise "NIF create_matrix/3 not implemented"
  end

  def matrix_pos(_matrix, _row, _column) do
    raise "NIF matrix_pos/3 not implemented"
  end

  def size_of_matrix(_matrix) do
    raise "NIF size_of_matrix/1 not implemented"
  end

  def matrix_to_lists(_matrix) do
    raise "NIF matrix_to_term/1 not implemented"
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

  def float_tensor(_float) do
    raise "NIF float_tensor/1 not implemented"
  end

  def float_tensor(_values, _dims) do
    raise "NIF float_tensor/2 not implemented"
  end

  def string_tensor(_string) do
    raise "NIF string_tensor/1 not implemented"
  end

  def tensor_datatype(_tensor) do
    raise "NIF tensor_datatype/1 not implemented"
  end

end
