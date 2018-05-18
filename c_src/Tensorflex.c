#include "erl_nif.h"
#include "c_api.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define BASE_STRING_LENGTH 255

void free_buffer(void* data, size_t length) {
  free(data);
}

ErlNifResourceType *graph_resource, *op_desc_resource, *tensor_resource, *session_resource, *op_resource, *buffer_resource, *status_resource, *graph_opts_resource;

void graph_destr(ErlNifEnv *env, void *res) {
  TF_DeleteGraph(*(TF_Graph **)res);
}

void graph_opts_destr(ErlNifEnv *env, void *res) {
  TF_DeleteImportGraphDefOptions(*(TF_ImportGraphDefOptions **)res);
}

void tensor_destr(ErlNifEnv *env, void *res) {
  TF_DeleteTensor(*(TF_Tensor **)res);
}

void status_destr(ErlNifEnv *env, void *res) {
  TF_DeleteStatus(*(TF_Status**)res);
}

void buffer_destr(ErlNifEnv *env, void *res) {
  TF_DeleteBuffer(*(TF_Buffer **)res);
}

void session_destr(ErlNifEnv *env, void *res) {
  TF_Status *status = TF_NewStatus();
  TF_DeleteSession(*(TF_Session **)res, status);
  if (TF_GetCode(status) != TF_OK) {
    fprintf(stderr, "Error: Cannot delete session!: %s\r\n", TF_Message(status));
  }
  TF_DeleteStatus(status);
}

void op_destr(ErlNifEnv *env, void *res) {}

void op_desc_destr(ErlNifEnv *env, void *res) {}

void tensor_deallocator(void* data, size_t len, void* arg) {
  fprintf(stderr, "free tensor %p\r\n", data);
  enif_free(data);
}

static ERL_NIF_TERM version(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  return enif_make_string(env, TF_Version() , ERL_NIF_LATIN1);
}

int res_loader(ErlNifEnv *env, void **priv_data, ERL_NIF_TERM load_info) {
  int flags = ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER;
  graph_resource = enif_open_resource_type(env, NULL, "graph", graph_destr, flags, NULL);
  op_desc_resource = enif_open_resource_type(env, NULL, "op_desc", op_desc_destr, flags, NULL);
  op_resource = enif_open_resource_type(env, NULL, "op", op_destr, flags, NULL);
  status_resource = enif_open_resource_type(env, NULL, "status", status_destr, flags, NULL);
  tensor_resource = enif_open_resource_type(env, NULL, "tensor", tensor_destr, flags, NULL);
  session_resource = enif_open_resource_type(env, NULL, "session", session_destr, flags, NULL);
  buffer_resource = enif_open_resource_type(env, NULL, "buffer", buffer_destr, flags, NULL);
  graph_opts_resource = enif_open_resource_type(env, NULL, "graph_opts",graph_opts_destr, flags, NULL);
  return 0;
}

static ERL_NIF_TERM string_constant(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  char buf[1024];
  enif_get_string(env, argv[0] , buf, 1024, ERL_NIF_LATIN1);
  return enif_make_string(env, buf, ERL_NIF_LATIN1);
}

static ERL_NIF_TERM new_import_graph_def_opts(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  TF_ImportGraphDefOptions **graph_opts_resource_alloc = enif_alloc_resource(graph_opts_resource, sizeof(TF_ImportGraphDefOptions *));
  TF_ImportGraphDefOptions *new_graph_opts = TF_NewImportGraphDefOptions();
  memcpy((void *) graph_opts_resource_alloc, (void *) &new_graph_opts, sizeof(TF_ImportGraphDefOptions *));
  ERL_NIF_TERM graph_opts = enif_make_resource(env, graph_opts_resource_alloc);
  enif_release_resource(graph_opts_resource_alloc);
  return graph_opts;
}

static ERL_NIF_TERM new_graph(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  TF_Graph **graph_resource_alloc = enif_alloc_resource(graph_resource, sizeof(TF_Graph *));
  TF_Graph *new_graph = TF_NewGraph();
  memcpy((void *) graph_resource_alloc, (void *) &new_graph, sizeof(TF_Graph *));
  ERL_NIF_TERM graph = enif_make_resource(env, graph_resource_alloc);
  enif_release_resource(graph_resource_alloc);
  return graph;
}

static ERL_NIF_TERM new_op(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
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

const char* error_to_string(TF_Status* status, char* error)
{
  switch(TF_GetCode(status))
    {
    case TF_CANCELLED: strcpy(error,"cancelled");
      break;
    case TF_UNKNOWN: strcpy(error,"unknown");
      break;
    case TF_INVALID_ARGUMENT: strcpy(error,"invalid_argument");
      break;
    case TF_DEADLINE_EXCEEDED: strcpy(error,"deadline_exceeded");
      break;
    case TF_NOT_FOUND: strcpy(error,"not_found");
      break;
    case TF_ALREADY_EXISTS: strcpy(error, "already_exists");
      break;
    case TF_PERMISSION_DENIED: strcpy(error,"permission_denied");
      break;
    case TF_UNAUTHENTICATED: strcpy(error,"unauthenticated");
      break;
    case TF_RESOURCE_EXHAUSTED: strcpy(error,"resource_exhausted");
      break;
    case TF_FAILED_PRECONDITION: strcpy(error,"failed_precondition");
      break;
    case TF_ABORTED: strcpy(error,"aborted");
      break;
    case TF_OUT_OF_RANGE: strcpy(error,"out_of_range");
      break;
    case TF_UNIMPLEMENTED: strcpy(error,"unimplemented");
      break;
    case TF_INTERNAL: strcpy(error,"internal");
      break;
    case TF_UNAVAILABLE: strcpy(error,"unavailable");
      break;
    case TF_DATA_LOSS: strcpy(error,"data_loss");
      break;
    default: strcpy(error,"unlisted_code");
    }
  
  return error;
}

static ERL_NIF_TERM read_graph(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  ErlNifBinary filepath;
  enif_inspect_binary(env,argv[0], &filepath);

  char* file = enif_alloc(filepath.size+1);
  memset(file, 0, filepath.size+1);
  memcpy(file, (void *) filepath.data, filepath.size);

  FILE *f = fopen(file, "rb");
  fseek(f, 0, SEEK_END);
  long fsize = ftell(f);
  fseek(f, 0, SEEK_SET);

  void* data = malloc(fsize);
  fread(data, fsize, 1, f);
  fclose(f);

  TF_Buffer* buf = TF_NewBuffer();
  buf->data = data;
  buf->length = fsize;
  buf->data_deallocator = free_buffer;

  TF_Status* status = TF_NewStatus();
  TF_ImportGraphDefOptions *graph_opts = TF_NewImportGraphDefOptions();
  TF_Graph *graph = TF_NewGraph();
  
  TF_GraphImportGraphDef(graph, buf, graph_opts, status);
  if (TF_GetCode(status) != TF_OK) {

    char error_str[BASE_STRING_LENGTH];
    error_to_string(status, error_str);
    return enif_make_tuple2(env,enif_make_atom(env,"error"),enif_make_atom(env,error_str));
  }
  else {
    fprintf(stderr, "Successfully imported graph\n");
  }

  TF_Graph **graph_resource_alloc = enif_alloc_resource(graph_resource, sizeof(TF_Graph *));
  memcpy((void *) graph_resource_alloc, (void *) &graph, sizeof(TF_Graph *));
  ERL_NIF_TERM loaded_graph = enif_make_resource(env, graph_resource_alloc);
  enif_release_resource(graph_resource_alloc);
  return loaded_graph;
  
}

static ERL_NIF_TERM get_graph_ops(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  TF_Graph **graph;
  enif_get_resource(env, argv[0], graph_resource, (void *) &graph);

  int n_ops = 0;
  size_t pos = 0;
  TF_Operation *op_count;
  while ((op_count = TF_GraphNextOperation(*graph, &pos)) != NULL) {
    n_ops++;
  }

  ERL_NIF_TERM *op_list;
  ERL_NIF_TERM op_list_eterm;
  TF_Operation *op_temp;
  ErlNifBinary erl_str;
  op_list = malloc(sizeof(ERL_NIF_TERM)*n_ops);
  pos = 0;
  
  char op_name[BASE_STRING_LENGTH];
  for(int i=0; i<n_ops; i++) {
    op_temp = TF_GraphNextOperation(*graph, &pos);
    strcpy(op_name, (char*) TF_OperationName(op_temp));
    enif_alloc_binary(strlen(op_name), &erl_str);
    memcpy(erl_str.data, op_name, strlen(op_name));
    op_list[i] = enif_make_binary(env,&erl_str);
  }

  op_list_eterm = enif_make_list_from_array(env, op_list, n_ops);
  free(op_list);
  return op_list_eterm;
}

static ERL_NIF_TERM create_and_run_sess(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
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
    { "version", 0, version },
    { "read_graph", 1, read_graph },
    { "get_graph_ops", 1, get_graph_ops },
  };

ERL_NIF_INIT(Elixir.Tensorflex, nif_funcs, res_loader, NULL, NULL, NULL)

