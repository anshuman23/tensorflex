#include "erl_nif.h"
#include "c_api.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <assert.h>

typedef struct
{
    unsigned nrows;
    unsigned ncols;
    double* data;
} Matrix;

typedef union
{
    void* vp;
    Matrix* p;
} mx_t;

#define POS(MX, ROW, COL) ((MX)->data[(ROW)* (MX)->ncols + (COL)])

#define BASE_STRING_LENGTH 255

static int get_number(ErlNifEnv* env, ERL_NIF_TERM term, double* dp);
static Matrix* alloc_matrix(ErlNifEnv* env, unsigned nrows, unsigned ncols);
static void matrix_destr(ErlNifEnv* env, void* obj);

static ErlNifResourceType* resource_type = NULL;


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

  ErlNifResourceType* rt = enif_open_resource_type(env, NULL, "matrix", matrix_destr, ERL_NIF_RT_CREATE, NULL);
    if (rt == NULL) {
	return -1;
    }
    assert(resource_type == NULL);
    resource_type = rt;

  return 0;
}


static ERL_NIF_TERM create_matrix(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    unsigned nrows, ncols;
    unsigned i, j;
    ERL_NIF_TERM list, row, ret;
    Matrix* mx = NULL;

    if (!enif_get_uint(env, argv[0], &nrows) || nrows < 1 ||
	!enif_get_uint(env, argv[1], &ncols) || ncols < 1) {

	goto badarg;
    }
    mx = alloc_matrix(env, nrows, ncols);
    list = argv[2];
    for (i = 0; i<nrows; i++) {
	if (!enif_get_list_cell(env, list, &row, &list)) {
	    goto badarg;
	}
	for (j = 0; j<ncols; j++) {
	    ERL_NIF_TERM v;
	    if (!enif_get_list_cell(env, row, &v, &row) ||
		!get_number(env, v, &POS(mx,i,j))) { 
		goto badarg;
	    }	    
	}
	if (!enif_is_empty_list(env, row)) {
	    goto badarg;
	}
    }
    if (!enif_is_empty_list(env, list)) {
	goto badarg;
    }

    ret = enif_make_resource(env, mx);
    enif_release_resource(mx);
    return ret;

badarg:
    if (mx != NULL) {
	enif_release_resource(mx);
    }
    return enif_make_badarg(env);
}


static ERL_NIF_TERM matrix_pos(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    mx_t mx;
    unsigned i, j;
    if (!enif_get_resource(env, argv[0], resource_type, &mx.vp) ||
	!enif_get_uint(env, argv[1], &i) || (--i >= mx.p->nrows) ||
	!enif_get_uint(env, argv[2], &j) || (--j >= mx.p->ncols)) {
	return enif_make_badarg(env);
    }
    return enif_make_double(env, POS(mx.p, i,j));
}

static ERL_NIF_TERM size_of_matrix(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    mx_t mx;
    if (!enif_get_resource(env, argv[0], resource_type, &mx.vp)) {
	return enif_make_badarg(env);
    }
    return enif_make_tuple2(env, enif_make_uint(env, mx.p->nrows),
			    enif_make_uint(env, mx.p->ncols));
}

static ERL_NIF_TERM matrix_to_lists(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    unsigned i, j;
    ERL_NIF_TERM res;
    mx_t mx;
    mx.p = NULL;

    if (!enif_get_resource(env, argv[0], resource_type, &mx.vp)) {
    	return enif_make_badarg(env);
    }
    res = enif_make_list(env, 0);
    for (i = mx.p->nrows; i-- > 0; ) {
	ERL_NIF_TERM row = enif_make_list(env, 0);
	for (j = mx.p->ncols; j-- > 0; ) {
	    row = enif_make_list_cell(env, enif_make_double(env, POS(mx.p,i,j)),
				      row);
	}
	res = enif_make_list_cell(env, row, res);
    }
    return res;
}

static int get_number(ErlNifEnv* env, ERL_NIF_TERM term, double* dp)
{
    long i;
    return enif_get_double(env, term, dp) || 
	(enif_get_long(env, term, &i) && (*dp=(double)i, 1));
}

static Matrix* alloc_matrix(ErlNifEnv* env, unsigned nrows, unsigned ncols)
{
    Matrix* mx = enif_alloc_resource(resource_type, sizeof(Matrix));
    mx->nrows = nrows;
    mx->ncols = ncols;
    mx->data = enif_alloc(nrows*ncols*sizeof(double));
    return mx;
}

static void matrix_destr(ErlNifEnv* env, void* obj)
{
    Matrix* mx = (Matrix*) obj;
    enif_free(mx->data);
    mx->data = NULL;
}

static ERL_NIF_TERM error_to_atom(ErlNifEnv *env, TF_Status* status)
{
  switch(TF_GetCode(status))
    {
    case TF_CANCELLED: return enif_make_atom(env,"cancelled");
      break;
    case TF_UNKNOWN: return enif_make_atom(env,"unknown");
      break;
    case TF_INVALID_ARGUMENT: return enif_make_atom(env,"invalid_argument");
      break;
    case TF_DEADLINE_EXCEEDED: return enif_make_atom(env,"deadline_exceeded");
      break;
    case TF_NOT_FOUND: return enif_make_atom(env,"not_found");
      break;
    case TF_ALREADY_EXISTS: return enif_make_atom(env, "already_exists");
      break;
    case TF_PERMISSION_DENIED: return enif_make_atom(env,"permission_denied");
      break;
    case TF_UNAUTHENTICATED: return enif_make_atom(env,"unauthenticated");
      break;
    case TF_RESOURCE_EXHAUSTED: return enif_make_atom(env,"resource_exhausted");
      break;
    case TF_FAILED_PRECONDITION: return enif_make_atom(env,"failed_precondition");
      break;
    case TF_ABORTED: return enif_make_atom(env,"aborted");
      break;
    case TF_OUT_OF_RANGE: return enif_make_atom(env,"out_of_range");
      break;
    case TF_UNIMPLEMENTED:return enif_make_atom(env,"unimplemented");
      break;
    case TF_INTERNAL: return enif_make_atom(env,"internal");
      break;
    case TF_UNAVAILABLE: return enif_make_atom(env,"unavailable");
      break;
    case TF_DATA_LOSS: return enif_make_atom(env,"data_loss");
      break;
    default: return enif_make_atom(env,"unlisted_code");
    }  
}

static ERL_NIF_TERM datatype_to_atom(ErlNifEnv *env, TF_DataType type)
{
  switch(type)
    {
    case TF_FLOAT: return enif_make_atom(env,"tf_float");
      break;
    case TF_DOUBLE: return enif_make_atom(env,"tf_double");
      break;
    case TF_INT32: return enif_make_atom(env,"tf_int32");
      break;
    case TF_UINT8: return enif_make_atom(env,"tf_uint8");
      break;
    case TF_INT16: return enif_make_atom(env,"tf_int16");
      break;
    case TF_INT8: return enif_make_atom(env, "tf_int8");
      break;
    case TF_STRING: return enif_make_atom(env,"tf_string");
      break;
    case TF_COMPLEX64: return enif_make_atom(env,"tf_complex64");
      break;
    case TF_INT64: return enif_make_atom(env,"tf_int64");
      break;
    case TF_BOOL: return enif_make_atom(env,"tf_bool");
      break;
    case TF_QINT8: return enif_make_atom(env,"tf_qint8");
      break;
    case TF_QUINT8: return enif_make_atom(env,"tf_quint8");
      break;
    case TF_QINT32:return enif_make_atom(env,"tf_qint32");
      break;
    case TF_BFLOAT16: return enif_make_atom(env,"tf_bfloat16");
      break;
    case TF_QINT16: return enif_make_atom(env,"tf_qint16");
      break;
    case TF_QUINT16: return enif_make_atom(env,"tf_quint16");
      break;
    case TF_UINT16: return enif_make_atom(env,"tf_uint16");
      break;
    case TF_COMPLEX128: return enif_make_atom(env,"tf_complex128");
      break;
    case TF_HALF: return enif_make_atom(env,"tf_half");
      break;
    case TF_RESOURCE: return enif_make_atom(env,"tf_resource");
      break;
    case TF_VARIANT: return enif_make_atom(env,"tf_variant");
      break;
    default: return enif_make_atom(env,"unlisted_datatype");
    }  
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
    return enif_make_tuple2(env, enif_make_atom(env,"error"), error_to_atom(env,status));
  }

  TF_Graph **graph_resource_alloc = enif_alloc_resource(graph_resource, sizeof(TF_Graph *));
  memcpy((void *) graph_resource_alloc, (void *) &graph, sizeof(TF_Graph *));
  ERL_NIF_TERM loaded_graph = enif_make_resource(env, graph_resource_alloc);
  enif_release_resource(graph_resource_alloc);
  return enif_make_tuple2(env, enif_make_atom(env,"ok"), loaded_graph);
  
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
  
  for(int i=0; i<n_ops; i++) {
    op_temp = TF_GraphNextOperation(*graph, &pos);
    enif_alloc_binary(strlen((char*) TF_OperationName(op_temp)), &erl_str);
    memcpy(erl_str.data, (char*) TF_OperationName(op_temp), strlen((char*) TF_OperationName(op_temp)));
    op_list[i] = enif_make_binary(env, &erl_str);
  }

  op_list_eterm = enif_make_list_from_array(env, op_list, n_ops);
  free(op_list);
  return op_list_eterm;
}


static ERL_NIF_TERM tensor_datatype(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  TF_Tensor **tensor;
  enif_get_resource(env, argv[0], tensor_resource, (void *) &tensor);
  TF_DataType type =  TF_TensorType(*tensor);
  return enif_make_tuple2(env, enif_make_atom(env,"ok"), datatype_to_atom(env, type));
}

static ERL_NIF_TERM string_tensor(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) 
{
  TF_Tensor *tensor;
  TF_Tensor **tensor_resource_alloc = enif_alloc_resource(tensor_resource, sizeof(TF_Tensor *));

  if (!(enif_is_binary(env, argv[0]))) {
	return enif_make_badarg(env);
  }
  ErlNifBinary str;
  enif_inspect_binary(env, argv[0], &str);

  TF_Status *status = TF_NewStatus();
  void *val = enif_alloc(str.size+1);
  memset(val, 0, str.size+1);
  TF_StringEncode((void *) str.data, str.size, val, str.size+1, status);

  if (TF_GetCode(status) != TF_OK) {
 	return enif_make_tuple2(env, enif_make_atom(env,"error"), error_to_atom(env,status));
  }

  tensor = TF_NewTensor(TF_STRING, 0, 0, val, str.size, tensor_deallocator, 0);
  memcpy((void *) tensor_resource_alloc, (void *) &tensor, sizeof(TF_Tensor *));
  ERL_NIF_TERM new_tensor = enif_make_resource(env, tensor_resource_alloc);
  enif_release_resource(tensor_resource_alloc);
  return enif_make_tuple2(env, enif_make_atom(env,"ok"), new_tensor);
}

static ERL_NIF_TERM float_tensor(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) 
{
  TF_Tensor *tensor;
  TF_Tensor **tensor_resource_alloc = enif_alloc_resource(tensor_resource, sizeof(TF_Tensor *));

  if (enif_is_number(env, argv[0])) {
    void *val = enif_alloc(sizeof(double));
    if (enif_get_double(env, argv[0], val)) {
      tensor = TF_NewTensor(TF_FLOAT, 0, 0, val, sizeof(double), tensor_deallocator, 0);
    } else return enif_make_badarg(env);
  }

  else {
    mx_t mx1, mx2;
    if (!enif_get_resource(env, argv[0], resource_type, &mx1.vp) || !enif_get_resource(env, argv[1], resource_type, &mx2.vp) || mx2.p->nrows > 1) {
	return enif_make_badarg(env);
    }

    int ndims = (int)(mx1.p->ncols);

    unsigned i,j;
    int64_t dims[mx2.p->ncols];
    int size_alloc = 1;
    for (i = 0; i < mx2.p->nrows; i++) {
	for (j = 0; j < mx2.p->ncols; j++) {
	    size_alloc = size_alloc * POS(mx2.p, i, j);
            dims[j] = POS(mx2.p, i, j);
	}
    }

    tensor = TF_NewTensor(TF_DOUBLE, dims, ndims, mx1.p->data, (size_alloc) * sizeof(double), tensor_deallocator, 0);

  }

  memcpy((void *) tensor_resource_alloc, (void *) &tensor, sizeof(TF_Tensor *));
  ERL_NIF_TERM new_tensor = enif_make_resource(env, tensor_resource_alloc);
  enif_release_resource(tensor_resource_alloc);
  return enif_make_tuple2(env, enif_make_atom(env,"ok"), new_tensor);
}



static ErlNifFunc nif_funcs[] =
  {
    {"create_matrix", 3, create_matrix},
    {"matrix_pos", 3, matrix_pos},
    {"size_of_matrix", 1, size_of_matrix},
    {"matrix_to_lists", 1, matrix_to_lists},
    { "version", 0, version },
    { "read_graph", 1, read_graph },
    { "get_graph_ops", 1, get_graph_ops },
    { "float_tensor", 2, float_tensor },
    { "float_tensor", 1, float_tensor },
    { "string_tensor", 1, string_tensor },
    { "tensor_datatype", 1, tensor_datatype },
  };

ERL_NIF_INIT(Elixir.Tensorflex, nif_funcs, res_loader, NULL, NULL, NULL)

