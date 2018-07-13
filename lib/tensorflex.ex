defmodule Tensorflex do

  alias Tensorflex.{NIFs, Graph, Tensor, Matrix}

  defp empty_list?([[]]), do: true
  defp empty_list?(list) when is_list(list) do
    false
  end
  
  def read_graph(filepath) do
    unless File.exists?(filepath) do
      raise ArgumentError, message: "Graph definition file does not exist"
    end

    unless (Path.extname(filepath) == ".pb") do
      raise ArgumentError, message: "File is not a protobuf .pb file"
    end
     
    {:ok, ref} = NIFs.read_graph(filepath)
    {:ok, %Graph{def: ref, name: filepath}}
  end

  def get_graph_ops(%Graph{def: ref, name: filepath}) do
    NIFs.get_graph_ops(ref)
  end

  def create_matrix(nrows, ncols, datalist) do
    if(nrows < 1 or ncols < 1) do
      raise ArgumentError, message: "Rows/Columns cannot be less than 1"
    end
    if(empty_list? datalist) do
      raise ArgumentError, message: "Data provided cannot be an empty list"
    end
    
    ref = NIFs.create_matrix(nrows, ncols, datalist)
    %Matrix{nrows: nrows, ncols: ncols, data: ref}
  end

  def matrix_pos(%Matrix{nrows: nrows, ncols: ncols, data: ref}, row, col) do
    if(row < 1 or col < 1) do
      raise ArgumentError, message: "Row/Column cannot be less than 1"
    end
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
    unless is_float(floatval) do
      raise ArgumentError, message: "Argument should be a float value for float64_tensor/1"
    end
    {:ok, ref} = NIFs.float64_tensor(floatval)
    {:ok, %Tensor{datatype: :tf_double, tensor: ref}}
  end

  def float32_tensor(%Matrix{nrows: val_rows, ncols: val_cols, data: val_ref}, %Matrix{nrows: dim_rows, ncols: dim_cols, data: dim_ref}) do
    {:ok, ref} = NIFs.float32_tensor(val_ref, dim_ref)
    {:ok, %Tensor{datatype: :tf_float, tensor: ref}} 
  end

  def float32_tensor(floatval) do
    unless is_float(floatval) do
      raise ArgumentError, message: "Argument should be a float value for float64_tensor/1"
    end
    {:ok, ref} = NIFs.float32_tensor(floatval)
    {:ok, %Tensor{datatype: :tf_float, tensor: ref}}
  end

  def string_tensor(stringval) do
    unless is_binary(stringval) do
      raise ArgumentError, message: "Argument should be a string value for string_tensor/1"
    end
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
    unless File.exists?(imagepath) do
      raise ArgumentError, message: "Image file does not exist"
    end

    unless (Path.extname(imagepath) == ".jpg" or Path.extname(imagepath) == ".jpeg") do
      raise ArgumentError, message: "File is not a JPEG image file"
    end
    
    {:ok, ref} = NIFs.load_image_as_tensor(imagepath)
    {:ok, %Tensor{datatype: :tf_uint8, tensor: ref}}
  end

  def run_session(%Graph{def: graphdef, name: filepath}, %Tensor{datatype: input_datatype, tensor: input_ref}, %Tensor{datatype: output_datatype, tensor: output_ref}, input_opname, output_opname) do
    NIFs.run_session(graphdef, input_ref, output_ref, input_opname, output_opname)
  end
  
end
