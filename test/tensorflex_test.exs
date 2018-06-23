defmodule TensorflexTest do
  use ExUnit.Case
  doctest Tensorflex

  test "Matrix Functionalities Test" do
    mat = Tensorflex.create_matrix(2,3, [[2.2,1.3,44.5],[5.5,6.1,3.333]])
    assert 5.5 = Tensorflex.matrix_pos(mat,2,1)
    assert {2,3} = Tensorflex.size_of_matrix(mat)
    assert [[2.2,1.3,44.5],[5.5,6.1,3.333]] = Tensorflex.matrix_to_lists(mat)
  end

  test "Tensor Functionalities Test" do
    dims = Tensorflex.create_matrix(1,3,[[1,1,3]])
    vals = Tensorflex.create_matrix(1,3,[[245,202,9]])
    assert {:ok, :tf_float} = Tensorflex.tensor_datatype(elem(Tensorflex.float32_tensor(vals,dims),1))
    assert {:ok, :tf_float} = Tensorflex.tensor_datatype(elem(Tensorflex.float32_tensor_alloc(dims),1))
    assert {:ok, :tf_float} = Tensorflex.tensor_datatype(elem(Tensorflex.float32_tensor(123.123),1))
    assert {:ok, :tf_string} = Tensorflex.tensor_datatype(elem(Tensorflex.string_tensor("123.123"),1))
    assert {:ok, :tf_double} = Tensorflex.tensor_datatype(elem(Tensorflex.float64_tensor(123.123),1))
    assert_raise ArgumentError, fn ->
      Tensorflex.float32_tensor("123.123")
    end
    assert_raise ArgumentError, fn ->
      Tensorflex.string_tensor(123.123)
    end
  end

  test "Graph Loading Test" do
    assert ["biases", "biases/read", "weights", "weights/read", "input", "MatMul", "add", "output"] = Tensorflex.get_graph_ops(elem(Tensorflex.read_graph("./test/graphdef_toy.pb"),1))
    assert ["biases2", "biases2/read", "weights2", "weights2/read", "biases1", "biases1/read", "weights1", "weights1/read", "input", "MatMul", "Add", "Relu", "MatMul_1", "Add_1", "output"] = Tensorflex.get_graph_ops(elem(Tensorflex.read_graph("./test/graphdef_iris.pb"),1))
    assert_raise ArgumentError, fn ->
      Tensorflex.read_graph("Makefile")
    end
    assert_raise ArgumentError, fn ->
      Tensorflex.read_graph("graph.txt")
    end
  end

  test "Session Test" do
    {:ok, graph} = Tensorflex.read_graph("./test/graphdef_toy.pb")
    in_vals = Tensorflex.create_matrix(3,3,[[1.0,1.0,1.0],[2.0,2.0,2.0],[3.0,3.0,3.0]])
    in_dims = Tensorflex.create_matrix(1,2,[[3,3]])
    {:ok, input_tensor} = Tensorflex.float32_tensor(in_vals, in_dims)
    out_dims = Tensorflex.create_matrix(1,2,[[3,2]])
    {:ok, output_tensor} = Tensorflex.float32_tensor_alloc(out_dims)
    assert [[56.349998474121094, 39.26000213623047], [109.69999694824219, 75.52000427246094], [163.04998779296875, 111.77999877929688]] = Tensorflex.run_session(graph, input_tensor, output_tensor, "input", "output")
  end
  
end



  
  
