defmodule TensorflexTest do
  use ExUnit.Case

  describe "core functionalities" do

    test "basic matrix functions check" do
      mat = Tensorflex.create_matrix(4,4,[[123,431,23,1],[1,2,3,4],[5,6,7,8],[768,564,44,5]])
      assert {4,4} = Tensorflex.size_of_matrix mat
      assert 44.0 = Tensorflex.matrix_pos(mat,4,3)
      assert [[123.0, 431.0, 23.0, 1.0],[1.0, 2.0, 3.0, 4.0],[5.0, 6.0, 7.0, 8.0],[768.0, 564.0, 44.0, 5.0]] = Tensorflex.matrix_to_lists mat
    end
    
    test "running session/graph loading check" do
      {:ok, graph} = Tensorflex.read_graph("./test/graphdef_toy.pb")
      assert ["input", "weights", "weights/read", "biases", "biases/read", "MatMul", "add", "output"] = Tensorflex.get_graph_ops graph
      in_vals = Tensorflex.create_matrix(3,3,[[1.0,1.0,1.0],[2.0,2.0,2.0],[3.0,3.0,3.0]])
      in_dims = Tensorflex.create_matrix(1,2,[[3,3]])
      {:ok, input_tensor} = Tensorflex.float32_tensor(in_vals, in_dims)
      out_dims = Tensorflex.create_matrix(1,2,[[3,2]])
      {:ok, output_tensor} = Tensorflex.float32_tensor_alloc(out_dims)
      assert [[56.349998474121094, 39.26000213623047], [109.69999694824219, 75.52000427246094], [163.04998779296875, 111.77999877929688]] = Tensorflex.run_session(graph, input_tensor, output_tensor, "input", "output")
    end

    test "CSV loading function check" do
      m_header = Tensorflex.load_csv_as_matrix("./test/sample2.csv", header: :true, delimiter: "-")
      assert [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]] = Tensorflex.matrix_to_lists m_header
      m_no_header = Tensorflex.load_csv_as_matrix("./test/sample1.csv", header: :false)
      assert [[1.0, 2.0, 3.0, 4.0, 5.0],[6.0, 7.0, 8.0, 9.0, 10.0],[11.0, 12.0, 13.0, 14.0, 15.0]] = Tensorflex.matrix_to_lists m_no_header

      assert_raise ArgumentError, fn ->
	_m = Tensorflex.load_csv_as_matrix("./test/sample1.csv", header: :no_header, delimiter: ",")
      end
    end

    test "float32 tensor functions check" do
      dims = Tensorflex.create_matrix(1,2,[[1,3]])
      vals = Tensorflex.create_matrix(1,3,[[245,202,9]])
      {:ok, tensor} = Tensorflex.float32_tensor vals,dims
      {:ok, :tf_float} = Tensorflex.tensor_datatype(tensor)
      {:ok, ftensor} = Tensorflex.float32_tensor 1234.1234
      {:ok, :tf_float} = Tensorflex.tensor_datatype(ftensor)
      {:ok, tensor_alloc} = Tensorflex.float32_tensor_alloc dims
      {:ok, :tf_float} = Tensorflex.tensor_datatype(tensor_alloc)

      assert_raise FunctionClauseError, fn ->
	Tensorflex.float32_tensor("123.123")
      end

      assert_raise FunctionClauseError, fn ->
	Tensorflex.float32_tensor(123)
      end
    end

    test "int32 tensor functions check" do
      dims = Tensorflex.create_matrix(1,2,[[1,3]])
      vals = Tensorflex.create_matrix(1,3,[[245,202,9]])
      {:ok, tensor} = Tensorflex.int32_tensor vals,dims
      {:ok, :tf_int32} = Tensorflex.tensor_datatype(tensor)
      {:ok, ftensor} = Tensorflex.int32_tensor 1234
      {:ok, :tf_int32} = Tensorflex.tensor_datatype(ftensor)
      {:ok, tensor_alloc} = Tensorflex.int32_tensor_alloc dims
      {:ok, :tf_int32} = Tensorflex.tensor_datatype(tensor_alloc)

      assert_raise FunctionClauseError, fn ->
	Tensorflex.int32_tensor("123.123")
      end

      assert_raise FunctionClauseError, fn ->
	Tensorflex.int32_tensor(123.123)
      end
    end    
  end
end
