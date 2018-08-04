defmodule TensorflexTest do
  use ExUnit.Case

  describe "matrix functionalities" do  
    test "matrix creation check" do
      assert [[2.2,1.3,44.5],[5.5,6.1,3.333]] = Tensorflex.create_matrix(2,3, [[2.2,1.3,44.5],[5.5,6.1,3.333]]) |> Tensorflex.matrix_to_lists
    end
    
    test "matrix to lists conversion check" do
      mat = Tensorflex.create_matrix(5,4,[[123,431,23,1],[1,2,3,4],[5,6,7,8],[768,564,44,5],[1,2,3,4]])
      assert [[123.0, 431.0, 23.0, 1.0],[1.0, 2.0, 3.0, 4.0],[5.0, 6.0, 7.0, 8.0],[768.0, 564.0, 44.0, 5.0],[1.0, 2.0, 3.0, 4.0]] = Tensorflex.matrix_to_lists mat
    end

    test "matrix access function check" do
      mat = Tensorflex.create_matrix(2,3, [[2.2,1.3,44.5],[5.5,6.1,3.333]])
      assert 5.5 = Tensorflex.matrix_pos(mat,2,1)
      assert 3.333 = Tensorflex.matrix_pos(mat,2,3)
    end

    test "get size of matrix" do
      assert {3,3} = Tensorflex.create_matrix(3,3, [[3.9,62,122],[2.2,1.3,44.5],[5.5,6.1,3.333]]) |> Tensorflex.size_of_matrix 
    end

    test "append new row to matrix function check" do
      mat = Tensorflex.create_matrix(4,4,[[123,431,23,1],[1,2,3,4],[5,6,7,8],[768,564,44,5]])
      mat = Tensorflex.append_to_matrix(mat, [[4.4,2,7,9.9]])
      assert {5,4} = Tensorflex.size_of_matrix mat
      assert 7.0 = Tensorflex.matrix_pos(mat,5,3)
    end

    test "add scalar to matrix check" do
      m = Tensorflex.create_matrix(2,3,[[1,2,3],[4,5,6]])
      m = Tensorflex.add_scalar_to_matrix(m, 5)
      assert [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]] = Tensorflex.matrix_to_lists m
    end

    test "subtract scalar from matrix check" do
      m = Tensorflex.create_matrix(2,3,[[1,2,3],[4,5,6]])
      m = Tensorflex.subtract_scalar_from_matrix m,3
      assert [[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]] = Tensorflex.matrix_to_lists m
    end

    test "multiply matrix with scalar check" do
      m = Tensorflex.create_matrix(2,3,[[1,2,3],[4,5,6]])
      m = Tensorflex.multiply_matrix_with_scalar m,5
      assert [[5.0, 10.0, 15.0], [20.0, 25.0, 30.0]] = Tensorflex.matrix_to_lists m
    end

    test "divide matrix with scalar check" do
      m = Tensorflex.create_matrix(2,3,[[1,2,3],[4,5,6]])
      m = Tensorflex.divide_matrix_by_scalar m,3
      assert [[0.3333333333333333, 0.6666666666666666, 1.0], [1.3333333333333333, 1.6666666666666667, 2.0]] = Tensorflex.matrix_to_lists m
    end

    test "add two matrices check" do
      m1 = Tensorflex.create_matrix(2,3,[[1,2,3],[4,5,6]])
      m2 = Tensorflex.create_matrix(2,3,[[4,5,6],[1,2,3]])
      m_added = Tensorflex.add_matrices m1,m2
      assert [[5.0, 7.0, 9.0], [5.0, 7.0, 9.0]] = Tensorflex.matrix_to_lists m_added
    end

    test "subtract two matrices check" do
      m1 = Tensorflex.create_matrix(2,3,[[1,2,3],[4,5,6]])
      m2 = Tensorflex.create_matrix(2,3,[[4,5,6],[1,2,3]])
      m_subtracted = Tensorflex.subtract_matrices m1,m2
      assert [[-3.0, -3.0, -3.0], [3.0, 3.0, 3.0]] = Tensorflex.matrix_to_lists m_subtracted
    end
    
  end

  describe "float32 tensor functionalities" do
    test "float32_tensor/2 tensor creation check" do
      dims = Tensorflex.create_matrix(1,2,[[1,3]])
      vals = Tensorflex.create_matrix(1,3,[[245,202,9]])
      {:ok, tensor} = Tensorflex.float32_tensor vals,dims
      {:ok, :tf_float} = Tensorflex.tensor_datatype(tensor)
    end

    test "float32_tensor/1 tensor creation check" do
      {:ok, tensor} = Tensorflex.float32_tensor 1234.1234
      {:ok, :tf_float} = Tensorflex.tensor_datatype(tensor)
    end

    test "float32_tensor_alloc/1 tensor creation check" do
      dims = Tensorflex.create_matrix(1,2,[[1,3]])
      {:ok, tensor} = Tensorflex.float32_tensor_alloc dims
      {:ok, :tf_float} = Tensorflex.tensor_datatype(tensor)
    end

    test "incorrect usage check" do
      assert_raise FunctionClauseError, fn ->
	Tensorflex.float32_tensor("123.123")
      end
      
      assert_raise FunctionClauseError, fn ->
	Tensorflex.float32_tensor(123)
      end
    end
  end

  describe "float64 tensor functionalities" do
    test "float64_tensor/2 tensor creation check" do
      dims = Tensorflex.create_matrix(1,2,[[1,3]])
      vals = Tensorflex.create_matrix(1,3,[[245,202,9]])
      {:ok, tensor} = Tensorflex.float64_tensor vals,dims
      {:ok, :tf_double} = Tensorflex.tensor_datatype(tensor)
    end

    test "float64_tensor/1 tensor creation check" do
      {:ok, tensor} = Tensorflex.float64_tensor 1234.1234
      {:ok, :tf_double} = Tensorflex.tensor_datatype(tensor)
    end

    test "float64_tensor_alloc/1 tensor creation check" do
      dims = Tensorflex.create_matrix(1,2,[[1,3]])
      {:ok, tensor} = Tensorflex.float64_tensor_alloc dims
      {:ok, :tf_double} = Tensorflex.tensor_datatype(tensor)
    end
    
    test "incorrect usage check" do
      assert_raise FunctionClauseError, fn ->
	Tensorflex.float64_tensor("123.123")
      end
      
      assert_raise FunctionClauseError, fn ->
	Tensorflex.float64_tensor(123)
      end
    end
  end

  describe "int32 tensor functionalities" do
    test "int32_tensor/2 tensor creation check" do
      dims = Tensorflex.create_matrix(1,2,[[1,3]])
      vals = Tensorflex.create_matrix(1,3,[[245,202,9]])
      {:ok, tensor} = Tensorflex.int32_tensor vals,dims
      {:ok, :tf_int32} = Tensorflex.tensor_datatype(tensor)
    end

    test "int32_tensor/1 tensor creation check" do
      {:ok, tensor} = Tensorflex.int32_tensor 1234
      {:ok, :tf_int32} = Tensorflex.tensor_datatype(tensor)
    end

    test "int32_tensor_alloc/1 tensor creation check" do
      dims = Tensorflex.create_matrix(1,2,[[1,3]])
      {:ok, tensor} = Tensorflex.int32_tensor_alloc dims
      {:ok, :tf_int32} = Tensorflex.tensor_datatype(tensor)
    end

    test "incorrect usage check" do
      assert_raise FunctionClauseError, fn ->
	Tensorflex.int32_tensor("123.123")
      end
      
      assert_raise FunctionClauseError, fn ->
	Tensorflex.int32_tensor(123.123)
      end
    end
  end

  describe "string tensor functionality" do
    test "string tensor creation check" do
      {:ok, tensor} = Tensorflex.string_tensor "123.123"
      {:ok, :tf_string} = Tensorflex.tensor_datatype tensor
    end

    test "incorrect usage check" do
      assert_raise FunctionClauseError, fn ->
	Tensorflex.string_tensor(123.123)
      end
      
      assert_raise FunctionClauseError, fn ->
	Tensorflex.string_tensor(123)
      end
    end
  end

  describe "graph loading and reading functionalities" do
    test "graph loading check" do
      {:ok, _graph_toy} = Tensorflex.read_graph "./test/graphdef_toy.pb"
      {:ok, _graph_iris} = Tensorflex.read_graph "./test/graphdef_iris.pb"
    end

    test "get all graph ops" do
      {:ok, graph_toy} = Tensorflex.read_graph "./test/graphdef_toy.pb"
      {:ok, graph_iris} = Tensorflex.read_graph "./test/graphdef_iris.pb"
      assert ["input", "weights", "weights/read", "biases", "biases/read", "MatMul", "add", "output"] = Tensorflex.get_graph_ops graph_toy
      assert ["input", "weights1", "weights1/read", "biases1", "biases1/read", "weights2", "weights2/read", "biases2", "biases2/read", "MatMul", "Add", "Relu", "MatMul_1", "Add_1", "output"] = Tensorflex.get_graph_ops graph_iris
    end

    test "incorrect usage check" do
      assert_raise ArgumentError, fn ->
	{:ok, _graph} = Tensorflex.read_graph "Makefile"
      end
      
      assert_raise ArgumentError, fn ->
	{:ok, _graph} = Tensorflex.read_graph "Makefile.pb"
      end
    end
  end

  describe "session functionality" do
    test "running session check" do
      {:ok, graph} = Tensorflex.read_graph("./test/graphdef_toy.pb")
      in_vals = Tensorflex.create_matrix(3,3,[[1.0,1.0,1.0],[2.0,2.0,2.0],[3.0,3.0,3.0]])
      in_dims = Tensorflex.create_matrix(1,2,[[3,3]])
      {:ok, input_tensor} = Tensorflex.float32_tensor(in_vals, in_dims)
      out_dims = Tensorflex.create_matrix(1,2,[[3,2]])
      {:ok, output_tensor} = Tensorflex.float32_tensor_alloc(out_dims)
      assert [[56.349998474121094, 39.26000213623047], [109.69999694824219, 75.52000427246094], [163.04998779296875, 111.77999877929688]] = Tensorflex.run_session(graph, input_tensor, output_tensor, "input", "output")
    end
  end

  describe "miscellaneous functionalities" do
    test "CSV-with-header loading function check" do
      m = Tensorflex.load_csv_as_matrix("./test/sample2.csv", header: :true, delimiter: "-")
      assert [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]] = Tensorflex.matrix_to_lists m
    end

    test "CSV-without-header loading function check" do
      m = Tensorflex.load_csv_as_matrix("./test/sample1.csv", header: :false)
      assert [[1.0, 2.0, 3.0, 4.0, 5.0],[6.0, 7.0, 8.0, 9.0, 10.0],[11.0, 12.0, 13.0, 14.0, 15.0]] = Tensorflex.matrix_to_lists m
    end

    test "CSV-to-matrix function incorrect usage check" do
      assert_raise ArgumentError, fn ->
	_m = Tensorflex.load_csv_as_matrix("./test/sample1.csv", header: :no_header, delimiter: ",")
      end
    end

    test "image-to-tensor loading function check" do
      {:ok, img_tensor} = Tensorflex.load_image_as_tensor("./test/cropped_panda.jpg")
      {:ok, :tf_uint8} = Tensorflex.tensor_datatype img_tensor
    end

    test "image-to-tensor function incorrect usage check" do
      assert_raise ArgumentError, fn -> 
	{:ok, _tensor} = Tensorflex.load_image_as_tensor("./test/sample1.csv")
      end
    end

    test "float64_tensor to matrix conversion check" do
      vals = Tensorflex.load_csv_as_matrix("./test/sample1.csv", header: :false)
      dims = Tensorflex.create_matrix(1,2,[[3,5]])
      {:ok, float64_tensor} = Tensorflex.float64_tensor vals,dims
      m_float64 = Tensorflex.tensor_to_matrix float64_tensor
      assert [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0, 15.0]] = Tensorflex.matrix_to_lists m_float64 
    end

    test "int32_tensor to matrix conversion check" do
      vals = Tensorflex.load_csv_as_matrix("./test/sample1.csv", header: :false)
      dims = Tensorflex.create_matrix(1,2,[[3,5]])
      {:ok, int32_tensor} = Tensorflex.int32_tensor vals,dims
      m_int32 = Tensorflex.tensor_to_matrix int32_tensor
      assert [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0, 15.0]] = Tensorflex.matrix_to_lists m_int32
    end

    test "float32_tensor to matrix conversion check" do
      vals = Tensorflex.load_csv_as_matrix("./test/sample1.csv", header: :false)
      dims = Tensorflex.create_matrix(1,2,[[3,5]])
      {:ok, float32_tensor} = Tensorflex.float32_tensor vals,dims
      m_float32 = Tensorflex.tensor_to_matrix float32_tensor
      assert [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0, 15.0]] = Tensorflex.matrix_to_lists m_float32
    end

    test "incorrect usage tensor-to-matrix check" do
      {:ok, tensor} = Tensorflex.load_image_as_tensor("./test/cropped_panda.jpg")
      assert_raise ArgumentError, fn -> 
	_m = Tensorflex.tensor_to_matrix tensor
      end
    end
  end
  
end
