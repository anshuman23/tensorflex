defmodule Tensorflex.NIFs do
  @moduledoc false
  
  @on_load :load_nifs

  def load_nifs do
    :erlang.load_nif(Application.app_dir(:tensorflex, "priv/Tensorflex"), 0)
  end

  def create_matrix(_nrows,_ncolumns, _list) do
    raise "NIF create_matrix/3 not implemented"
  end

  def matrix_pos(_matrix, _row, _column) do
    raise "NIF matrix_pos/3 not implemented"
  end

  def append_to_matrix(_matrix, _datalist) do
    raise "NIF append_to_matrix/2 not implemented"
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

  def float64_tensor(_float) do
    raise "NIF float_tensor64/1 not implemented"
  end

  def float64_tensor(_values, _dims) do
    raise "NIF float_tensor64/2 not implemented"
  end

  def float32_tensor(_float) do
    raise "NIF float_tensor32/1 not implemented"
  end

  def float32_tensor(_values, _dims) do
    raise "NIF float_tensor32/2 not implemented"
  end

  def int32_tensor(_values, _dims) do
    raise "NIF int32_tensor/2 not implemented"
  end

  def int32_tensor(_int) do
    raise "NIF int32_tensor/1 not implemented"
  end

  def string_tensor(_string) do
    raise "NIF string_tensor/1 not implemented"
  end

  def tensor_datatype(_tensor) do
    raise "NIF tensor_datatype/1 not implemented"
  end

  def int32_tensor_alloc(_dims) do
    raise "NIF int32_tensor_alloc/1 not implemented"
  end

  def float32_tensor_alloc(_dims) do
    raise "NIF float32_tensor_alloc/1 not implemented"
  end

  def float64_tensor_alloc(_dims) do
    raise "NIF float64_tensor_alloc/1 not implemented"
  end

  def load_image_as_tensor(_imagepath) do
    raise "NIF load_image_as_tensor/1 not implemented"
  end
  
  def load_csv_as_matrix(_filepath, _header, _delimiter) do
    raise "NIF load_csv_as_matrix/3 not implemented"
  end

  def run_session(_graph, _input_tensor, _output_tensor, _input_opname, _output_opname) do
    raise "NIF run_session/5 not implemented"
  end

  def add_scalar_to_matrix(_mat, _scalar) do
    raise "NIF add_scalar_to_matrix/2 not implemented"
  end

  def subtract_scalar_from_matrix(_mat, _scalar) do
    raise "NIF subtract_scalar_from_matrix/2 not implemented"
  end

  def multiply_matrix_with_scalar(_mat, _scalar) do
    raise "NIF multiply_matrix_with_scalar/2 not implemented"
  end

  def divide_matrix_by_scalar(_mat, _scalar) do
    raise "NIF divide_matrix_by_scalar/2 not implemented"
  end

  def add_matrices(_mat1, _mat2) do
    raise "NIF add_matrices/2 not implemented"
  end

  def subtract_matrices(_mat1, _mat2) do
    raise "NIF subtract_matrices/2 not implemented"
  end

  def tensor_to_matrix(_tensor) do
    raise "NIF tensor_to_matrix/1 not implemented"
  end

  def binary_to_matrix(_binary, _nrows, _ncols) do
    raise "NIF binary_to_matrix/3 not implemented"
  end

  def matrix_to_float32_tensor(_matrix, _dims) do
    raise "NIF matrix_to_float32_tensor/2 not implemented"
  end

end
