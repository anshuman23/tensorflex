#include "erl_nif.h"
#include "c_api.h"
#include <string.h>
#include <stdio.h>

ErlNifResourceType *graph_resource, *op_desc_resource;

void graph_destr(ErlNifEnv *env, void *res) {
  TF_DeleteGraph(*(TF_Graph **)res);
}

void op_desc_destr(ErlNifEnv *env, void *res) {}

static ERL_NIF_TERM tf_version(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  return enif_make_string(env, TF_Version() , ERL_NIF_LATIN1);
}

int res_loader(ErlNifEnv *env, void **priv_data, ERL_NIF_TERM load_info) {
  int flags = ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER;
  graph_resource = enif_open_resource_type(env, NULL, "graph", graph_destr, flags, NULL);
  op_desc_resource = enif_open_resource_type(env, NULL, "op_desc", op_desc_destr, flags, NULL);
  return 0;
}

static ERL_NIF_TERM tf_string_constant(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  char buf[1024];
  enif_get_string(env, argv[0] , buf, 1024, ERL_NIF_LATIN1);
  return enif_make_string(env, buf, ERL_NIF_LATIN1);
}

static ERL_NIF_TERM tf_new_graph(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  TF_Graph **graph_resource_alloc = enif_alloc_resource(graph_resource, sizeof(TF_Graph *));
  TF_Graph *new_graph = TF_NewGraph();
  memcpy((void *) graph_resource_alloc, (void *) &new_graph, sizeof(TF_Graph *));
  ERL_NIF_TERM graph = enif_make_resource(env, graph_resource_alloc);
  enif_release_resource(graph_resource_alloc);
  return graph;
}


static ERL_NIF_TERM tf_new_op(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  TF_Graph **graph;
  enif_get_resource(env, argv[0], graph_resource, (void *) &graph);
  ErlNifBinary op_str;
  enif_inspect_binary(env, argv[1], &op_str);
  char* op = enif_alloc(op_str.size+1);
  memset(op, 0, op_str.size+1);
  memcpy(op, (void *) op_str.data, op_str.size);
  ErlNifBinary name;
  enif_inspect_binary(env, argv[2], &name);
  char* label = enif_alloc(name.size+1);
  memset(label, 0, name.size+1);
  memcpy(label, (void *) name.data, name.size);

  TF_OperationDescription **op_desc_resource_alloc = enif_alloc_resource(op_desc_resource, sizeof(TF_OperationDescription *));
  TF_OperationDescription *new_op_desc = TF_NewOperation(*graph, op, label);
  memcpy((void *) op_desc_resource_alloc, (void *) &new_op_desc, sizeof(TF_OperationDescription *));
  ERL_NIF_TERM op_desc = enif_make_resource(env, op_desc_resource_alloc);
  enif_release_resource(op_desc_resource_alloc);
  return op_desc;
}

static ERL_NIF_TERM tf_create_and_run_sess(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  TF_Graph **graph;
  enif_get_resource(env, argv[0], graph_resource, (void *) &graph);
  fprintf(stderr, "%s\n", "\n=> [INFO] Loaded Graph correctly\n");
  TF_OperationDescription **op_desc;
  enif_get_resource(env, argv[1], op_desc_resource, (void *) &op_desc);
  fprintf(stderr, "%s\n", "\n=> [INFO] Loaded Operation Description correctly");
  
  char buf[1024];
  enif_get_string(env, argv[2] , buf, 1024, ERL_NIF_LATIN1);
  TF_Tensor * tensor = TF_AllocateTensor(TF_STRING, 0, 0, 8 + TF_StringEncodedSize( strlen(buf)));
  TF_Status *status = TF_NewStatus();
  TF_SessionOptions * options = TF_NewSessionOptions();
  TF_Session * session = TF_NewSession(*graph, options, status);
  TF_Tensor * output_tensor;
  TF_Operation * operation;
  struct TF_Output output;

  TF_StringEncode(buf, strlen(buf), 8 + (char *) TF_TensorData(tensor), TF_StringEncodedSize(strlen(buf)), status);
  memset(TF_TensorData(tensor), 0, 8);
  TF_SetAttrTensor(*op_desc, "value", tensor, status);
  TF_SetAttrType(*op_desc, "dtype", TF_TensorType(tensor));
  operation = TF_FinishOperation(*op_desc, status);
  output.oper = operation;
  output.index = 0;

  TF_SessionRun( session, 0,
		 0, 0, 0, 
		 &output, &output_tensor, 1,  
		 &operation, 1, 
		 0, status );

  fprintf(stderr, "%s\n", "\n=> [INFO] Session Run Complete");

  TF_CloseSession(session, status);
  TF_DeleteSession(session, status);
  TF_DeleteStatus(status);
  TF_DeleteSessionOptions(options);
  return enif_make_string(env, ((char *) TF_TensorData(output_tensor))+9 , ERL_NIF_LATIN1);
}

static ErlNifFunc nif_funcs[] =
  {
    { "tf_version", 0, tf_version },
    { "tf_new_graph", 0, tf_new_graph },
    { "tf_new_op", 3, tf_new_op },
    { "tf_string_constant", 1, tf_string_constant },
    { "tf_create_and_run_sess", 3, tf_create_and_run_sess }
  };

ERL_NIF_INIT(Elixir.TensorflEx, nif_funcs, res_loader, NULL, NULL, NULL)

