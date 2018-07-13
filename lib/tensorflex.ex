defmodule Tensorflex do

  import Tensorflex.NIFs
  import Tensorflex.Graph
  import Tensorflex.Matrix
  import Tensorflex.Tensor
  
  def read_graph(filepath) do
    {:ok, ref} = Tensorflex.NIFs.read_graph(filepath)
    {:ok, %Tensorflex.Graph{def: ref, name: filepath}}
  end

  def get_graph_ops(%Tensorflex.Graph{def: ref, name: filepath}) do
    Tensorflex.NIFs.get_graph_ops(ref)
  end

  def create_matrix(nrows, ncols, datalist) do
    ref = Tensorflex.NIFs.create_matrix(nrows, ncols, datalist)
    %Tensorflex.Matrix{nrows: nrows, ncols: ncols, data: ref}
  end

  def matrix_pos(%Tensorflex.Matrix{nrows: nrows, ncols: ncols, data: ref}, row, col) do
    Tensorflex.NIFs.matrix_pos(ref, row, col)
  end

  def size_of_matrix(%Tensorflex.Matrix{nrows: nrows, ncols: ncols, data: ref}) do
    {nrows, ncols}
  end

  def matrix_to_lists(%Tensorflex.Matrix{nrows: nrows, ncols: ncols, data: ref}) do
    Tensorflex.NIFs.matrix_to_lists(ref)
  end

  def float64_tensor(%Tensorflex.Matrix{nrows: val_rows, ncols: val_cols, data: val_ref}, %Tensorflex.Matrix{nrows: dim_rows, ncols: dim_cols, data: dim_ref}) do
    {:ok, ref} = Tensorflex.NIFs.float64_tensor(val_ref, dim_ref)
    {:ok, %Tensorflex.Tensor{datatype: :tf_double, tensor: ref}} 
  end

  def float64_tensor(floatval) do
    {:ok, ref} = Tensorflex.NIFs.float64_tensor(floatval)
    {:ok, %Tensorflex.Tensor{datatype: :tf_double, tensor: ref}}
  end

  def float32_tensor(%Tensorflex.Matrix{nrows: val_rows, ncols: val_cols, data: val_ref}, %Tensorflex.Matrix{nrows: dim_rows, ncols: dim_cols, data: dim_ref}) do
    {:ok, ref} = Tensorflex.NIFs.float32_tensor(val_ref, dim_ref)
    {:ok, %Tensorflex.Tensor{datatype: :tf_float, tensor: ref}} 
  end

  def float32_tensor(floatval) do
    {:ok, ref} = Tensorflex.NIFs.float32_tensor(floatval)
    {:ok, %Tensorflex.Tensor{datatype: :tf_float, tensor: ref}}
  end

  def string_tensor(stringval) do
    {:ok, ref} = Tensorflex.NIFs.string_tensor(stringval)
    {:ok, %Tensorflex.Tensor{datatype: :tf_string, tensor: ref}}
  end

  def float32_tensor_alloc(%Tensorflex.Matrix{nrows: dim_rows, ncols: dim_cols, data: dim_ref}) do
    {:ok, ref} = Tensorflex.NIFs.float32_tensor_alloc(dim_ref)
    {:ok, %Tensorflex.Tensor{datatype: :tf_float, tensor: ref}} 
  end

  def float64_tensor_alloc(%Tensorflex.Matrix{nrows: dim_rows, ncols: dim_cols, data: dim_ref}) do
    {:ok, ref} = Tensorflex.NIFs.float64_tensor_alloc(dim_ref)
    {:ok, %Tensorflex.Tensor{datatype: :tf_double, tensor: ref}} 
  end

  def tensor_datatype(%Tensorflex.Tensor{datatype: datatype, tensor: ref}) do
    {:ok, datatype}
  end

  def load_image_as_tensor(imagepath) do
    {:ok, ref} = Tensorflex.NIFs.load_image_as_tensor(imagepath)
    {:ok, %Tensorflex.Tensor{datatype: :tf_uint8, tensor: ref}}
  end

  def run_session(%Tensorflex.Graph{def: graphdef, name: filepath}, %Tensorflex.Tensor{datatype: input_datatype, tensor: input_ref}, %Tensorflex.Tensor{datatype: output_datatype, tensor: output_ref}, input_opname, output_opname) do
    Tensorflex.NIFs.run_session(graphdef, input_ref, output_ref, input_opname, output_opname)
  end
  
end
