defmodule Tensorflex do

  alias Tensorflex.{NIFs, Graph, Tensor, Matrix}
  
  def read_graph(filepath) do
    {:ok, ref} = NIFs.read_graph(filepath)
    {:ok, %Graph{def: ref, name: filepath}}
  end

  def get_graph_ops(%Graph{def: ref, name: filepath}) do
    NIFs.get_graph_ops(ref)
  end

  def create_matrix(nrows, ncols, datalist) do
    ref = NIFs.create_matrix(nrows, ncols, datalist)
    %Matrix{nrows: nrows, ncols: ncols, data: ref}
  end

  def matrix_pos(%Matrix{nrows: nrows, ncols: ncols, data: ref}, row, col) do
    NIFs.matrix_pos(ref, row, col)
  end

  def size_of_matrix(%Matrix{nrows: nrows, ncols: ncols, data: ref}) do
    {nrows, ncols}
  end

  def matrix_to_lists(%Matrix{nrows: nrows, ncols: ncols, data: ref}) do
    NIFs.matrix_to_lists(ref)
  end

  def float64_tensor(%Matrix{nrows: val_rows, ncols: val_cols, data: val_ref}, %Matrix{nrows: dim_rows, ncols: dim_cols, data: dim_ref}) do
    {:ok, ref} = NIFs.float64_tensor(val_ref, dim_ref)
    {:ok, %Tensor{datatype: :tf_double, tensor: ref}} 
  end

  def float64_tensor(floatval) do
    {:ok, ref} = NIFs.float64_tensor(floatval)
    {:ok, %Tensor{datatype: :tf_double, tensor: ref}}
  end

  def float32_tensor(%Matrix{nrows: val_rows, ncols: val_cols, data: val_ref}, %Matrix{nrows: dim_rows, ncols: dim_cols, data: dim_ref}) do
    {:ok, ref} = NIFs.float32_tensor(val_ref, dim_ref)
    {:ok, %Tensor{datatype: :tf_float, tensor: ref}} 
  end

  def float32_tensor(floatval) do
    {:ok, ref} = NIFs.float32_tensor(floatval)
    {:ok, %Tensor{datatype: :tf_float, tensor: ref}}
  end

  def string_tensor(stringval) do
    {:ok, ref} = NIFs.string_tensor(stringval)
    {:ok, %Tensor{datatype: :tf_string, tensor: ref}}
  end

  def float32_tensor_alloc(%Matrix{nrows: dim_rows, ncols: dim_cols, data: dim_ref}) do
    {:ok, ref} = NIFs.float32_tensor_alloc(dim_ref)
    {:ok, %Tensor{datatype: :tf_float, tensor: ref}} 
  end

  def float64_tensor_alloc(%Matrix{nrows: dim_rows, ncols: dim_cols, data: dim_ref}) do
    {:ok, ref} = NIFs.float64_tensor_alloc(dim_ref)
    {:ok, %Tensor{datatype: :tf_double, tensor: ref}} 
  end

  def tensor_datatype(%Tensor{datatype: datatype, tensor: ref}) do
    {:ok, datatype}
  end

  def load_image_as_tensor(imagepath) do
    {:ok, ref} = NIFs.load_image_as_tensor(imagepath)
    {:ok, %Tensor{datatype: :tf_uint8, tensor: ref}}
  end

  def run_session(%Graph{def: graphdef, name: filepath}, %Tensor{datatype: input_datatype, tensor: input_ref}, %Tensor{datatype: output_datatype, tensor: output_ref}, input_opname, output_opname) do
    NIFs.run_session(graphdef, input_ref, output_ref, input_opname, output_opname)
  end
  
end
