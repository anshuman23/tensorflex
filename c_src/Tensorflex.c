#include "c_api.h"
#include "erl_nif.h"
#include <assert.h>
#include <stdio.h>
#include <jpeglib.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  unsigned nrows;
  unsigned ncols;
  double *data;
} Matrix;

typedef union {
  void *vp;
  Matrix *p;
} mx_t;

#define POS(MX, ROW, COL) ((MX)->data[(ROW) * (MX)->ncols + (COL)])
#define BUF_SIZE 500000

static int get_number(ErlNifEnv *env, ERL_NIF_TERM term, double *dp);
static Matrix *alloc_matrix(ErlNifEnv *env, unsigned nrows, unsigned ncols);
static void matrix_destr(ErlNifEnv *env, void *obj);

static ErlNifResourceType *resource_type = NULL;

void free_buffer(void *data, size_t length) { free(data); }

ErlNifResourceType *graph_resource, *op_desc_resource, *tensor_resource,
    *session_resource, *op_resource, *buffer_resource, *status_resource,
    *graph_opts_resource;

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
  TF_DeleteStatus(*(TF_Status **)res);
}

void buffer_destr(ErlNifEnv *env, void *res) {
  TF_DeleteBuffer(*(TF_Buffer **)res);
}

void session_destr(ErlNifEnv *env, void *res) {
  TF_Status *status = TF_NewStatus();
  TF_DeleteSession(*(TF_Session **)res, status);
  if (TF_GetCode(status) != TF_OK) {
    fprintf(stderr, "Error: Cannot delete session!: %s\r\n",
            TF_Message(status));
  }
  TF_DeleteStatus(status);
}

void op_destr(ErlNifEnv *env, void *res) {}

void op_desc_destr(ErlNifEnv *env, void *res) {}

void tensor_deallocator(void *data, size_t len, void *arg) { enif_free(data); }

static ERL_NIF_TERM version(ErlNifEnv *env, int argc,
                            const ERL_NIF_TERM argv[]) {
  return enif_make_string(env, TF_Version(), ERL_NIF_LATIN1);
}

int res_loader(ErlNifEnv *env, void **priv_data, ERL_NIF_TERM load_info) {
  int flags = ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER;
  graph_resource =
      enif_open_resource_type(env, NULL, "graph", graph_destr, flags, NULL);
  op_desc_resource =
      enif_open_resource_type(env, NULL, "op_desc", op_desc_destr, flags, NULL);
  op_resource = enif_open_resource_type(env, NULL, "op", op_destr, flags, NULL);
  status_resource =
      enif_open_resource_type(env, NULL, "status", status_destr, flags, NULL);
  tensor_resource =
      enif_open_resource_type(env, NULL, "tensor", tensor_destr, flags, NULL);
  session_resource =
      enif_open_resource_type(env, NULL, "session", session_destr, flags, NULL);
  buffer_resource =
      enif_open_resource_type(env, NULL, "buffer", buffer_destr, flags, NULL);
  graph_opts_resource = enif_open_resource_type(env, NULL, "graph_opts",
                                                graph_opts_destr, flags, NULL);

  ErlNifResourceType *rt = enif_open_resource_type(
      env, NULL, "matrix", matrix_destr, ERL_NIF_RT_CREATE, NULL);
  if (rt == NULL) {
    return -1;
  }
  assert(resource_type == NULL);
  resource_type = rt;

  return 0;
}

static ERL_NIF_TERM create_matrix(ErlNifEnv *env, int argc,
                                  const ERL_NIF_TERM argv[]) {
  unsigned nrows, ncols;
  unsigned i, j;
  ERL_NIF_TERM list, row, ret;
  Matrix *mx = NULL;

  if (!enif_get_uint(env, argv[0], &nrows) || nrows < 1 ||
      !enif_get_uint(env, argv[1], &ncols) || ncols < 1) {

    goto badarg;
  }
  mx = alloc_matrix(env, nrows, ncols);
  list = argv[2];
  for (i = 0; i < nrows; i++) {
    if (!enif_get_list_cell(env, list, &row, &list)) {
      goto badarg;
    }
    for (j = 0; j < ncols; j++) {
      ERL_NIF_TERM v;
      if (!enif_get_list_cell(env, row, &v, &row) ||
          !get_number(env, v, &POS(mx, i, j))) {
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

static ERL_NIF_TERM matrix_pos(ErlNifEnv *env, int argc,
                               const ERL_NIF_TERM argv[]) {
  mx_t mx;
  unsigned i, j;
  if (!enif_get_resource(env, argv[0], resource_type, &mx.vp) ||
      !enif_get_uint(env, argv[1], &i) || (--i >= mx.p->nrows) ||
      !enif_get_uint(env, argv[2], &j) || (--j >= mx.p->ncols)) {
    return enif_make_badarg(env);
  }
  return enif_make_double(env, POS(mx.p, i, j));
}

static ERL_NIF_TERM size_of_matrix(ErlNifEnv *env, int argc,
                                   const ERL_NIF_TERM argv[]) {
  mx_t mx;
  if (!enif_get_resource(env, argv[0], resource_type, &mx.vp)) {
    return enif_make_badarg(env);
  }
  return enif_make_tuple2(env, enif_make_uint(env, mx.p->nrows),
                          enif_make_uint(env, mx.p->ncols));
}

static ERL_NIF_TERM matrix_to_lists(ErlNifEnv *env, int argc,
                                    const ERL_NIF_TERM argv[]) {
  unsigned i, j;
  ERL_NIF_TERM res;
  mx_t mx;
  mx.p = NULL;

  if (!enif_get_resource(env, argv[0], resource_type, &mx.vp)) {
    return enif_make_badarg(env);
  }
  res = enif_make_list(env, 0);
  for (i = mx.p->nrows; i-- > 0;) {
    ERL_NIF_TERM row = enif_make_list(env, 0);
    for (j = mx.p->ncols; j-- > 0;) {
      row =
          enif_make_list_cell(env, enif_make_double(env, POS(mx.p, i, j)), row);
    }
    res = enif_make_list_cell(env, row, res);
  }
  return res;
}

static ERL_NIF_TERM append_to_matrix(ErlNifEnv *env, int argc,
                                     const ERL_NIF_TERM argv[]) {
  Matrix *mx_ret = NULL;
  unsigned j, it_i, it_j;
  unsigned nrows, ncols;
  ERL_NIF_TERM list, row, ret;
  mx_t mx;

  if (!enif_get_resource(env, argv[0], resource_type, &mx.vp))
    return enif_make_badarg(env);

  nrows = mx.p->nrows;
  ncols = mx.p->ncols;

  mx_ret = alloc_matrix(env, nrows + 1, ncols);
  for (it_i = 0; it_i < nrows; it_i++) {
    for (it_j = 0; it_j < ncols; it_j++) {
      POS(mx_ret, it_i, it_j) = POS(mx.p, it_i, it_j);
    }
  }

  list = argv[1];
  if (!enif_get_list_cell(env, list, &row, &list)) {
    goto badarg;
  }
  for (j = 0; j < ncols; j++) {
    ERL_NIF_TERM v;
    if (!enif_get_list_cell(env, row, &v, &row) ||
        !get_number(env, v, &POS(mx_ret, nrows, j))) {
      goto badarg;
    }
  }
  if (!enif_is_empty_list(env, row)) {
    goto badarg;
  }

  if (!enif_is_empty_list(env, list)) {
    goto badarg;
  }

  ret = enif_make_resource(env, mx_ret);
  enif_release_resource(mx_ret);
  return ret;

badarg:
  if (mx.p != NULL) {
    enif_release_resource(mx.p);
  }
  return enif_make_badarg(env);
}

static int get_number(ErlNifEnv *env, ERL_NIF_TERM term, double *dp) {
  long i;
  return enif_get_double(env, term, dp) ||
         (enif_get_long(env, term, &i) && (*dp = (double)i, 1));
}

static Matrix *alloc_matrix(ErlNifEnv *env, unsigned nrows, unsigned ncols) {
  Matrix *mx = enif_alloc_resource(resource_type, sizeof(Matrix));
  mx->nrows = nrows;
  mx->ncols = ncols;
  mx->data = enif_alloc(nrows * ncols * sizeof(double));
  return mx;
}

static void matrix_destr(ErlNifEnv *env, void *obj) {
  Matrix *mx = (Matrix *)obj;
  enif_free(mx->data);
  mx->data = NULL;
}

static ERL_NIF_TERM error_to_atom(ErlNifEnv *env, TF_Status *status) {
  switch (TF_GetCode(status)) {
  case TF_CANCELLED:
    return enif_make_atom(env, "cancelled");
    break;
  case TF_UNKNOWN:
    return enif_make_atom(env, "unknown");
    break;
  case TF_INVALID_ARGUMENT:
    return enif_make_atom(env, "invalid_argument");
    break;
  case TF_DEADLINE_EXCEEDED:
    return enif_make_atom(env, "deadline_exceeded");
    break;
  case TF_NOT_FOUND:
    return enif_make_atom(env, "not_found");
    break;
  case TF_ALREADY_EXISTS:
    return enif_make_atom(env, "already_exists");
    break;
  case TF_PERMISSION_DENIED:
    return enif_make_atom(env, "permission_denied");
    break;
  case TF_UNAUTHENTICATED:
    return enif_make_atom(env, "unauthenticated");
    break;
  case TF_RESOURCE_EXHAUSTED:
    return enif_make_atom(env, "resource_exhausted");
    break;
  case TF_FAILED_PRECONDITION:
    return enif_make_atom(env, "failed_precondition");
    break;
  case TF_ABORTED:
    return enif_make_atom(env, "aborted");
    break;
  case TF_OUT_OF_RANGE:
    return enif_make_atom(env, "out_of_range");
    break;
  case TF_UNIMPLEMENTED:
    return enif_make_atom(env, "unimplemented");
    break;
  case TF_INTERNAL:
    return enif_make_atom(env, "internal");
    break;
  case TF_UNAVAILABLE:
    return enif_make_atom(env, "unavailable");
    break;
  case TF_DATA_LOSS:
    return enif_make_atom(env, "data_loss");
    break;
  default:
    return enif_make_atom(env, "unlisted_code");
  }
}

static ERL_NIF_TERM datatype_to_atom(ErlNifEnv *env, TF_DataType type) {
  switch (type) {
  case TF_FLOAT:
    return enif_make_atom(env, "tf_float");
    break;
  case TF_DOUBLE:
    return enif_make_atom(env, "tf_double");
    break;
  case TF_INT32:
    return enif_make_atom(env, "tf_int32");
    break;
  case TF_UINT8:
    return enif_make_atom(env, "tf_uint8");
    break;
  case TF_INT16:
    return enif_make_atom(env, "tf_int16");
    break;
  case TF_INT8:
    return enif_make_atom(env, "tf_int8");
    break;
  case TF_STRING:
    return enif_make_atom(env, "tf_string");
    break;
  case TF_COMPLEX64:
    return enif_make_atom(env, "tf_complex64");
    break;
  case TF_INT64:
    return enif_make_atom(env, "tf_int64");
    break;
  case TF_BOOL:
    return enif_make_atom(env, "tf_bool");
    break;
  case TF_QINT8:
    return enif_make_atom(env, "tf_qint8");
    break;
  case TF_QUINT8:
    return enif_make_atom(env, "tf_quint8");
    break;
  case TF_QINT32:
    return enif_make_atom(env, "tf_qint32");
    break;
  case TF_BFLOAT16:
    return enif_make_atom(env, "tf_bfloat16");
    break;
  case TF_QINT16:
    return enif_make_atom(env, "tf_qint16");
    break;
  case TF_QUINT16:
    return enif_make_atom(env, "tf_quint16");
    break;
  case TF_UINT16:
    return enif_make_atom(env, "tf_uint16");
    break;
  case TF_COMPLEX128:
    return enif_make_atom(env, "tf_complex128");
    break;
  case TF_HALF:
    return enif_make_atom(env, "tf_half");
    break;
  case TF_RESOURCE:
    return enif_make_atom(env, "tf_resource");
    break;
  case TF_VARIANT:
    return enif_make_atom(env, "tf_variant");
    break;
  default:
    return enif_make_atom(env, "unlisted_datatype");
  }
}

static ERL_NIF_TERM read_graph(ErlNifEnv *env, int argc,
                               const ERL_NIF_TERM argv[]) {
  ErlNifBinary filepath;
  enif_inspect_binary(env, argv[0], &filepath);

  char *file = enif_alloc(filepath.size + 1);
  memset(file, 0, filepath.size + 1);
  memcpy(file, (void *)filepath.data, filepath.size);

  const char *dot = strrchr(file, '.');
  if (!dot || dot == file)
    return enif_make_badarg(env);
  if (strcmp((dot + 1), "pb") != 0)
    return enif_make_badarg(env);

  FILE *f = fopen(file, "rb");
  fseek(f, 0, SEEK_END);
  long fsize = ftell(f);
  fseek(f, 0, SEEK_SET);

  void *data = malloc(fsize);
  fread(data, fsize, 1, f);
  fclose(f);

  TF_Buffer *buf = TF_NewBuffer();
  buf->data = data;
  buf->length = fsize;
  buf->data_deallocator = free_buffer;

  TF_Status *status = TF_NewStatus();
  TF_ImportGraphDefOptions *graph_opts = TF_NewImportGraphDefOptions();
  TF_Graph *graph = TF_NewGraph();

  TF_GraphImportGraphDef(graph, buf, graph_opts, status);
  if (TF_GetCode(status) != TF_OK) {
    return enif_make_tuple2(env, enif_make_atom(env, "error"),
                            error_to_atom(env, status));
  }

  TF_Graph **graph_resource_alloc =
      enif_alloc_resource(graph_resource, sizeof(TF_Graph *));
  memcpy((void *)graph_resource_alloc, (void *)&graph, sizeof(TF_Graph *));
  ERL_NIF_TERM loaded_graph = enif_make_resource(env, graph_resource_alloc);
  enif_release_resource(graph_resource_alloc);
  return enif_make_tuple2(env, enif_make_atom(env, "ok"), loaded_graph);
}

static ERL_NIF_TERM get_graph_ops(ErlNifEnv *env, int argc,
                                  const ERL_NIF_TERM argv[]) {
  TF_Graph **graph;
  enif_get_resource(env, argv[0], graph_resource, (void *)&graph);

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
  op_list = malloc(sizeof(ERL_NIF_TERM) * n_ops);
  pos = 0;

  for (int i = 0; i < n_ops; i++) {
    op_temp = TF_GraphNextOperation(*graph, &pos);
    enif_alloc_binary(strlen((char *)TF_OperationName(op_temp)), &erl_str);
    memcpy(erl_str.data, (char *)TF_OperationName(op_temp),
           strlen((char *)TF_OperationName(op_temp)));
    op_list[i] = enif_make_binary(env, &erl_str);
  }

  op_list_eterm = enif_make_list_from_array(env, op_list, n_ops);
  free(op_list);
  return op_list_eterm;
}

static ERL_NIF_TERM tensor_datatype(ErlNifEnv *env, int argc,
                                    const ERL_NIF_TERM argv[]) {
  TF_Tensor **tensor;
  enif_get_resource(env, argv[0], tensor_resource, (void *)&tensor);
  TF_DataType type = TF_TensorType(*tensor);
  return enif_make_tuple2(env, enif_make_atom(env, "ok"),
                          datatype_to_atom(env, type));
}

static ERL_NIF_TERM string_tensor(ErlNifEnv *env, int argc,
                                  const ERL_NIF_TERM argv[]) {
  TF_Tensor *tensor;
  TF_Tensor **tensor_resource_alloc =
      enif_alloc_resource(tensor_resource, sizeof(TF_Tensor *));

  if (!(enif_is_binary(env, argv[0]))) {
    return enif_make_badarg(env);
  }
  ErlNifBinary str;
  enif_inspect_binary(env, argv[0], &str);

  TF_Status *status = TF_NewStatus();
  void *val = enif_alloc(str.size + 1);
  memset(val, 0, str.size + 1);
  TF_StringEncode((void *)str.data, str.size, val, str.size + 1, status);

  if (TF_GetCode(status) != TF_OK) {
    return enif_make_tuple2(env, enif_make_atom(env, "error"),
                            error_to_atom(env, status));
  }

  tensor = TF_NewTensor(TF_STRING, 0, 0, val, str.size, tensor_deallocator, 0);
  memcpy((void *)tensor_resource_alloc, (void *)&tensor, sizeof(TF_Tensor *));
  ERL_NIF_TERM new_tensor = enif_make_resource(env, tensor_resource_alloc);
  enif_release_resource(tensor_resource_alloc);
  return enif_make_tuple2(env, enif_make_atom(env, "ok"), new_tensor);
}

static ERL_NIF_TERM float64_tensor(ErlNifEnv *env, int argc,
                                   const ERL_NIF_TERM argv[]) {
  TF_Tensor *tensor;
  TF_Tensor **tensor_resource_alloc =
      enif_alloc_resource(tensor_resource, sizeof(TF_Tensor *));

  if (enif_is_number(env, argv[0])) {
    void *val = enif_alloc(sizeof(double));
    if (enif_get_double(env, argv[0], val)) {
      tensor = TF_NewTensor(TF_DOUBLE, 0, 0, val, sizeof(double),
                            tensor_deallocator, 0);
    } else
      return enif_make_badarg(env);
  }

  else {
    mx_t mx1, mx2;
    if (!enif_get_resource(env, argv[0], resource_type, &mx1.vp) ||
        !enif_get_resource(env, argv[1], resource_type, &mx2.vp) ||
        mx2.p->nrows > 1) {
      return enif_make_badarg(env);
    }

    int ndims = (int)(mx2.p->ncols);

    unsigned i, j;
    int64_t dims[mx2.p->ncols];
    int size_alloc = 1;
    for (i = 0; i < mx2.p->nrows; i++) {
      for (j = 0; j < mx2.p->ncols; j++) {
        size_alloc = size_alloc * POS(mx2.p, i, j);
        dims[j] = POS(mx2.p, i, j);
      }
    }

    double *data = enif_alloc((mx1.p->nrows) * (mx1.p->ncols) * sizeof(double));
    for (i = 0; i < mx1.p->nrows; i++) {
      for (j = 0; j < mx1.p->ncols; j++) {
        data[(i) * (mx1.p->ncols) + (j)] = (double)POS(mx1.p, i, j);
      }
    }

    tensor = TF_NewTensor(TF_DOUBLE, dims, ndims, data,
                          (size_alloc) * sizeof(double), tensor_deallocator, 0);
  }

  memcpy((void *)tensor_resource_alloc, (void *)&tensor, sizeof(TF_Tensor *));
  ERL_NIF_TERM new_tensor = enif_make_resource(env, tensor_resource_alloc);
  enif_release_resource(tensor_resource_alloc);
  return enif_make_tuple2(env, enif_make_atom(env, "ok"), new_tensor);
}

static ERL_NIF_TERM float32_tensor(ErlNifEnv *env, int argc,
                                   const ERL_NIF_TERM argv[]) {
  TF_Tensor *tensor;
  TF_Tensor **tensor_resource_alloc =
      enif_alloc_resource(tensor_resource, sizeof(TF_Tensor *));

  if (enif_is_number(env, argv[0])) {
    void *val = enif_alloc(sizeof(float));
    if (enif_get_double(env, argv[0], val)) {
      tensor = TF_NewTensor(TF_FLOAT, 0, 0, val, sizeof(float),
                            tensor_deallocator, 0);
    } else
      return enif_make_badarg(env);
  }

  else {
    mx_t mx1, mx2;
    if (!enif_get_resource(env, argv[0], resource_type, &mx1.vp) ||
        !enif_get_resource(env, argv[1], resource_type, &mx2.vp) ||
        mx2.p->nrows > 1) {
      return enif_make_badarg(env);
    }

    int ndims = (int)(mx2.p->ncols);

    unsigned i, j;
    int64_t dims[mx2.p->ncols];
    int size_alloc = 1;
    for (i = 0; i < mx2.p->nrows; i++) {
      for (j = 0; j < mx2.p->ncols; j++) {
        size_alloc = size_alloc * POS(mx2.p, i, j);
        dims[j] = POS(mx2.p, i, j);
      }
    }

    float *data = enif_alloc((mx1.p->nrows) * (mx1.p->ncols) * sizeof(float));
    for (i = 0; i < mx1.p->nrows; i++) {
      for (j = 0; j < mx1.p->ncols; j++) {
        data[(i) * (mx1.p->ncols) + (j)] = (float)POS(mx1.p, i, j);
      }
    }

    tensor = TF_NewTensor(TF_FLOAT, dims, ndims, data,
                          (size_alloc) * sizeof(float), tensor_deallocator, 0);
  }

  memcpy((void *)tensor_resource_alloc, (void *)&tensor, sizeof(TF_Tensor *));
  ERL_NIF_TERM new_tensor = enif_make_resource(env, tensor_resource_alloc);
  enif_release_resource(tensor_resource_alloc);
  return enif_make_tuple2(env, enif_make_atom(env, "ok"), new_tensor);
}

static ERL_NIF_TERM int32_tensor(ErlNifEnv *env, int argc,
                                 const ERL_NIF_TERM argv[]) {
  TF_Tensor *tensor;
  TF_Tensor **tensor_resource_alloc =
      enif_alloc_resource(tensor_resource, sizeof(TF_Tensor *));

  if (enif_is_number(env, argv[0])) {
    void *val = enif_alloc(sizeof(int32_t));
    if (enif_get_int(env, argv[0], val)) {
      tensor = TF_NewTensor(TF_INT32, 0, 0, val, sizeof(int32_t),
                            tensor_deallocator, 0);
    } else
      return enif_make_badarg(env);
  }

  else {
    mx_t mx1, mx2;
    if (!enif_get_resource(env, argv[0], resource_type, &mx1.vp) ||
        !enif_get_resource(env, argv[1], resource_type, &mx2.vp) ||
        mx2.p->nrows > 1) {
      return enif_make_badarg(env);
    }

    int ndims = (int)(mx2.p->ncols);

    unsigned i, j;
    int64_t dims[mx2.p->ncols];
    int size_alloc = 1;
    for (i = 0; i < mx2.p->nrows; i++) {
      for (j = 0; j < mx2.p->ncols; j++) {
        size_alloc = size_alloc * POS(mx2.p, i, j);
        dims[j] = POS(mx2.p, i, j);
      }
    }

    int32_t *data =
        enif_alloc((mx1.p->nrows) * (mx1.p->ncols) * sizeof(int32_t));
    for (i = 0; i < mx1.p->nrows; i++) {
      for (j = 0; j < mx1.p->ncols; j++) {
        data[(i) * (mx1.p->ncols) + (j)] = (int32_t)POS(mx1.p, i, j);
      }
    }

    tensor =
        TF_NewTensor(TF_INT32, dims, ndims, data,
                     (size_alloc) * sizeof(int32_t), tensor_deallocator, 0);
  }

  memcpy((void *)tensor_resource_alloc, (void *)&tensor, sizeof(TF_Tensor *));
  ERL_NIF_TERM new_tensor = enif_make_resource(env, tensor_resource_alloc);
  enif_release_resource(tensor_resource_alloc);
  return enif_make_tuple2(env, enif_make_atom(env, "ok"), new_tensor);
}

static ERL_NIF_TERM allocate_tensor(ErlNifEnv *env, int argc,
                                    const ERL_NIF_TERM argv[], char *datatype) {
  TF_Tensor *tensor;
  TF_Tensor **tensor_resource_alloc =
      enif_alloc_resource(tensor_resource, sizeof(TF_Tensor *));

  mx_t mx;
  if (!enif_get_resource(env, argv[0], resource_type, &mx.vp) ||
      mx.p->nrows > 1) {
    return enif_make_badarg(env);
  }

  int ndims = (int)(mx.p->ncols);
  unsigned i, j;
  int64_t dims[mx.p->ncols];
  int size_alloc = 1;
  for (i = 0; i < mx.p->nrows; i++) {
    for (j = 0; j < mx.p->ncols; j++) {
      size_alloc = size_alloc * POS(mx.p, i, j);
      dims[j] = POS(mx.p, i, j);
    }
  }

  if (strcmp(datatype, "TF_FLOAT") == 0) {
    tensor =
        TF_AllocateTensor(TF_FLOAT, dims, ndims, (size_alloc) * sizeof(float));
  } else if (strcmp(datatype, "TF_DOUBLE") == 0) {
    tensor = TF_AllocateTensor(TF_DOUBLE, dims, ndims,
                               (size_alloc) * sizeof(double));
  } else if (strcmp(datatype, "TF_INT32") == 0) {
    tensor = TF_AllocateTensor(TF_INT32, dims, ndims,
                               (size_alloc) * sizeof(int32_t));
  } else
    return enif_make_badarg(env);

  memcpy((void *)tensor_resource_alloc, (void *)&tensor, sizeof(TF_Tensor *));
  ERL_NIF_TERM new_tensor = enif_make_resource(env, tensor_resource_alloc);
  enif_release_resource(tensor_resource_alloc);
  return enif_make_tuple2(env, enif_make_atom(env, "ok"), new_tensor);
}

static ERL_NIF_TERM int32_tensor_alloc(ErlNifEnv *env, int argc,
                                       const ERL_NIF_TERM argv[]) {
  return allocate_tensor(env, argc, argv, "TF_INT32");
}

static ERL_NIF_TERM float32_tensor_alloc(ErlNifEnv *env, int argc,
                                         const ERL_NIF_TERM argv[]) {
  return allocate_tensor(env, argc, argv, "TF_FLOAT");
}

static ERL_NIF_TERM float64_tensor_alloc(ErlNifEnv *env, int argc,
                                         const ERL_NIF_TERM argv[]) {
  return allocate_tensor(env, argc, argv, "TF_DOUBLE");
}

static ERL_NIF_TERM run_session(ErlNifEnv *env, int argc,
                                const ERL_NIF_TERM argv[]) {
  TF_Graph **graph;
  enif_get_resource(env, argv[0], graph_resource, (void *)&graph);

  TF_Tensor **input_tensor;
  enif_get_resource(env, argv[1], tensor_resource, (void *)&input_tensor);

  TF_Tensor **output_tensor;
  enif_get_resource(env, argv[2], tensor_resource, (void *)&output_tensor);

  ErlNifBinary input_opname_bin;
  enif_inspect_binary(env, argv[3], &input_opname_bin);
  char *input_opname = enif_alloc(input_opname_bin.size + 1);
  memset(input_opname, 0, input_opname_bin.size + 1);
  memcpy(input_opname, (void *)input_opname_bin.data, input_opname_bin.size);

  ErlNifBinary output_opname_bin;
  enif_inspect_binary(env, argv[4], &output_opname_bin);
  char *output_opname = enif_alloc(output_opname_bin.size + 1);
  memset(output_opname, 0, output_opname_bin.size + 1);
  memcpy(output_opname, (void *)output_opname_bin.data, output_opname_bin.size);

  TF_Operation *input_op = TF_GraphOperationByName(*graph, input_opname);
  TF_Output input_op_o = {input_op, 0};
  TF_Operation *output_op = TF_GraphOperationByName(*graph, output_opname);
  TF_Output output_op_o = {output_op, 0};

  TF_Status *status = TF_NewStatus();
  TF_SessionOptions *sess_opts = TF_NewSessionOptions();
  TF_Session *session = TF_NewSession(*graph, sess_opts, status);
  assert(TF_GetCode(status) == TF_OK);

  TF_SessionRun(session, NULL, &input_op_o, &(*input_tensor), 1, &output_op_o,
                &(*output_tensor), 1, NULL, 0, NULL, status);

  ERL_NIF_TERM *data_list, *data_list_eterm, data_list_of_lists;
  data_list = malloc(sizeof(ERL_NIF_TERM) *
                     TF_Dim(*output_tensor, (TF_NumDims(*output_tensor) - 1)));
  data_list_eterm =
      malloc(sizeof(ERL_NIF_TERM) * ((int)(TF_Dim(*output_tensor, 0))));
  float *data = (float *)(TF_TensorData(*output_tensor));

  for (int j = 0; j < (int)(TF_Dim(*output_tensor, 0)); j++) {
    for (int i = 0;
         i < TF_Dim(*output_tensor, (TF_NumDims(*output_tensor) - 1)); i++) {
      data_list[i] = enif_make_double(env, *data++);
    }

    data_list_eterm[j] = enif_make_list_from_array(
        env, data_list,
        TF_Dim(*output_tensor, (TF_NumDims(*output_tensor) - 1)));
  }

  data_list_of_lists = enif_make_list_from_array(
      env, data_list_eterm, (int)(TF_Dim(*output_tensor, 0)));
  free(data_list);
  free(data_list_eterm);
  TF_CloseSession(session, status);
  TF_DeleteSession(session, status);
  TF_DeleteSessionOptions(sess_opts);
  TF_DeleteStatus(status);
  return data_list_of_lists;
}

static ERL_NIF_TERM load_image_as_tensor(ErlNifEnv *env, int argc,
                                         const ERL_NIF_TERM argv[]) {
  TF_Tensor *tensor;
  TF_Tensor **tensor_resource_alloc =
      enif_alloc_resource(tensor_resource, sizeof(TF_Tensor *));

  int error_check;
  unsigned long input_size;
  unsigned char *input_img;
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  unsigned long output_size;
  unsigned char *output;
  int row_stride, width, height, num_pixels;

  ErlNifBinary filepath;
  enif_inspect_binary(env, argv[0], &filepath);

  char *file = enif_alloc(filepath.size + 1);
  memset(file, 0, filepath.size + 1);
  memcpy(file, (void *)filepath.data, filepath.size);

  const char *dot = strrchr(file, '.');
  if (!dot || dot == file)
    return enif_make_badarg(env);
  if (!((strcmp((dot + 1), "jpg") == 0) || (strcmp((dot + 1), "jpeg") == 0)))
    return enif_make_badarg(env);

  FILE *f = fopen(file, "rb");
  fseek(f, 0, SEEK_END);
  input_size = ftell(f);
  fseek(f, 0, SEEK_SET);

  input_img = (unsigned char *)malloc(input_size);
  fread(input_img, input_size, 1, f);
  fclose(f);

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);
  jpeg_mem_src(&cinfo, input_img, input_size);
  error_check = jpeg_read_header(&cinfo, TRUE);

  if (error_check != 1)
    return enif_make_badarg(env);
  jpeg_start_decompress(&cinfo);
  width = cinfo.output_width;
  height = cinfo.output_height;
  num_pixels = cinfo.output_components;
  output_size = width * height * num_pixels;
  output = (unsigned char *)malloc(output_size);
  row_stride = width * num_pixels;

  while (cinfo.output_scanline < cinfo.output_height) {
    unsigned char *buf[1];
    buf[0] = output + (cinfo.output_scanline) * row_stride;
    jpeg_read_scanlines(&cinfo, buf, 1);
  }

  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  free(input_img);

  const int size_alloc = output_size * sizeof(unsigned char);
  int64_t dims[3] = {width, height, num_pixels};

  uint8_t *data = enif_alloc(output_size * sizeof(uint8_t));
  for (int it = 0; it < output_size; it++) {
    data[it] = (uint8_t)output[it];
  }

  tensor =
      TF_NewTensor(TF_UINT8, dims, 3, data, size_alloc, tensor_deallocator, 0);
  memcpy((void *)tensor_resource_alloc, (void *)&tensor, sizeof(TF_Tensor *));
  ERL_NIF_TERM new_tensor = enif_make_resource(env, tensor_resource_alloc);
  enif_release_resource(tensor_resource_alloc);
  return enif_make_tuple2(env, enif_make_atom(env, "ok"), new_tensor);
}

static ERL_NIF_TERM load_csv_as_matrix(ErlNifEnv *env, int argc,
                                       const ERL_NIF_TERM argv[]) {
  ERL_NIF_TERM mat_ret;
  ErlNifBinary filepath;
  enif_inspect_binary(env, argv[0], &filepath);
  char *file = enif_alloc(filepath.size + 1);
  memset(file, 0, filepath.size + 1);
  memcpy(file, (void *)filepath.data, filepath.size);
  char buf_init[BUF_SIZE], buf[BUF_SIZE];
  char *val_init, *line_init, *val, *line;

  unsigned int header_atom_len;
  enif_get_atom_length(env, argv[1], &header_atom_len, ERL_NIF_LATIN1);
  char *header_atom = (char *)enif_alloc(header_atom_len + 1);
  enif_get_atom(env, argv[1], header_atom, header_atom_len + 1, ERL_NIF_LATIN1);

  ErlNifBinary delimiter;
  enif_inspect_binary(env, argv[2], &delimiter);
  char *delimiter_str = enif_alloc(delimiter.size + 1);
  memset(delimiter_str, 0, delimiter.size + 1);
  memcpy(delimiter_str, (void *)delimiter.data, delimiter.size);

  FILE *f_init = fopen(file, "rb");
  unsigned i = 0, j = 0;
  while ((line_init = fgets(buf_init, sizeof(buf_init), f_init)) != NULL) {
    j = 0;
    val_init = strtok(line_init, delimiter_str);
    while (val_init != NULL) {
      val_init = strtok(NULL, delimiter_str);
      j++;
    }
    i++;
  }
  fclose(f_init);

  int flag = 0;
  if (strcmp(header_atom, "true") == 0) {
    i--;
    flag = 1;
  }

  mx_t mx;
  mx.p = alloc_matrix(env, i, j);
  FILE *f = fopen(file, "rb");
  i = 0;
  while ((line = fgets(buf, sizeof(buf), f)) != NULL) {
    j = 0;
    val = strtok(line, delimiter_str);
    while (val != NULL) {
      if (flag == 0) {
        POS(mx.p, i, j) = atof(val);
        j++;
      }
      val = strtok(NULL, delimiter_str);
    }

    if (flag == 1) {
      flag = 0;
      i--;
    }
    i++;
  }
  fclose(f);

  mat_ret = enif_make_resource(env, mx.p);
  enif_release_resource(mx.p);
  return mat_ret;
}

static ERL_NIF_TERM add_scalar_to_matrix(ErlNifEnv *env, int argc,
                                         const ERL_NIF_TERM argv[]) {
  ERL_NIF_TERM ret;
  unsigned i, j;
  mx_t mx, mx_ret;
  mx.p = NULL;
  mx_ret.p = NULL;

  if (!enif_get_resource(env, argv[0], resource_type, &mx.vp)) {
    return enif_make_badarg(env);
  }

  double scalar = 0.0;
  if (!enif_get_double(env, argv[1], &scalar)) {
    return enif_make_badarg(env);
  }

  mx_ret.p = alloc_matrix(env, mx.p->nrows, mx.p->ncols);
  for (i = 0; i < mx.p->nrows; i++) {
    for (j = 0; j < mx.p->ncols; j++) {
      POS(mx_ret.p, i, j) = POS(mx.p, i, j) + scalar;
    }
  }

  ret = enif_make_resource(env, mx_ret.p);
  enif_release_resource(mx_ret.p);
  return ret;
}

static ERL_NIF_TERM subtract_scalar_from_matrix(ErlNifEnv *env, int argc,
                                                const ERL_NIF_TERM argv[]) {
  ERL_NIF_TERM ret;
  unsigned i, j;
  mx_t mx, mx_ret;
  mx.p = NULL;
  mx_ret.p = NULL;

  if (!enif_get_resource(env, argv[0], resource_type, &mx.vp)) {
    return enif_make_badarg(env);
  }

  double scalar = 0.0;
  if (!enif_get_double(env, argv[1], &scalar)) {
    return enif_make_badarg(env);
  }

  mx_ret.p = alloc_matrix(env, mx.p->nrows, mx.p->ncols);
  for (i = 0; i < mx.p->nrows; i++) {
    for (j = 0; j < mx.p->ncols; j++) {
      POS(mx_ret.p, i, j) = POS(mx.p, i, j) - scalar;
    }
  }

  ret = enif_make_resource(env, mx_ret.p);
  enif_release_resource(mx_ret.p);
  return ret;
}

static ERL_NIF_TERM multiply_matrix_with_scalar(ErlNifEnv *env, int argc,
                                                const ERL_NIF_TERM argv[]) {
  ERL_NIF_TERM ret;
  unsigned i, j;
  mx_t mx, mx_ret;
  mx.p = NULL;
  mx_ret.p = NULL;

  if (!enif_get_resource(env, argv[0], resource_type, &mx.vp)) {
    return enif_make_badarg(env);
  }

  double scalar = 1.0;
  if (!enif_get_double(env, argv[1], &scalar)) {
    return enif_make_badarg(env);
  }

  mx_ret.p = alloc_matrix(env, mx.p->nrows, mx.p->ncols);
  for (i = 0; i < mx.p->nrows; i++) {
    for (j = 0; j < mx.p->ncols; j++) {
      POS(mx_ret.p, i, j) = POS(mx.p, i, j) * scalar;
    }
  }

  ret = enif_make_resource(env, mx_ret.p);
  enif_release_resource(mx_ret.p);
  return ret;
}

static ERL_NIF_TERM divide_matrix_by_scalar(ErlNifEnv *env, int argc,
                                            const ERL_NIF_TERM argv[]) {
  ERL_NIF_TERM ret;
  unsigned i, j;
  mx_t mx, mx_ret;
  mx.p = NULL;
  mx_ret.p = NULL;

  if (!enif_get_resource(env, argv[0], resource_type, &mx.vp)) {
    return enif_make_badarg(env);
  }

  double scalar = 1.0;
  ;
  if ((!enif_get_double(env, argv[1], &scalar)) || (scalar == 0.0)) {
    return enif_make_badarg(env);
  }

  mx_ret.p = alloc_matrix(env, mx.p->nrows, mx.p->ncols);
  for (i = 0; i < mx.p->nrows; i++) {
    for (j = 0; j < mx.p->ncols; j++) {
      POS(mx_ret.p, i, j) = POS(mx.p, i, j) / scalar;
    }
  }

  ret = enif_make_resource(env, mx_ret.p);
  enif_release_resource(mx_ret.p);
  return ret;
}

static ERL_NIF_TERM add_matrices(ErlNifEnv *env, int argc,
                                 const ERL_NIF_TERM argv[]) {
  unsigned i, j;
  ERL_NIF_TERM ret;
  mx_t mx1, mx2, mx;
  mx1.p = NULL;
  mx2.p = NULL;
  mx.p = NULL;

  if (!enif_get_resource(env, argv[0], resource_type, &mx1.vp) ||
      !enif_get_resource(env, argv[1], resource_type, &mx2.vp) ||
      mx1.p->nrows != mx2.p->nrows || mx1.p->ncols != mx2.p->ncols) {
    return enif_make_badarg(env);
  }
  mx.p = alloc_matrix(env, mx1.p->nrows, mx2.p->ncols);
  for (i = 0; i < mx1.p->nrows; i++) {
    for (j = 0; j < mx2.p->ncols; j++) {
      POS(mx.p, i, j) = POS(mx1.p, i, j) + POS(mx2.p, i, j);
    }
  }
  ret = enif_make_resource(env, mx.p);
  enif_release_resource(mx.p);
  return ret;
}

static ERL_NIF_TERM subtract_matrices(ErlNifEnv *env, int argc,
                                      const ERL_NIF_TERM argv[]) {
  unsigned i, j;
  ERL_NIF_TERM ret;
  mx_t mx1, mx2, mx;
  mx1.p = NULL;
  mx2.p = NULL;
  mx.p = NULL;

  if (!enif_get_resource(env, argv[0], resource_type, &mx1.vp) ||
      !enif_get_resource(env, argv[1], resource_type, &mx2.vp) ||
      mx1.p->nrows != mx2.p->nrows || mx1.p->ncols != mx2.p->ncols) {
    return enif_make_badarg(env);
  }
  mx.p = alloc_matrix(env, mx1.p->nrows, mx2.p->ncols);
  for (i = 0; i < mx1.p->nrows; i++) {
    for (j = 0; j < mx2.p->ncols; j++) {
      POS(mx.p, i, j) = POS(mx1.p, i, j) - POS(mx2.p, i, j);
    }
  }
  ret = enif_make_resource(env, mx.p);
  enif_release_resource(mx.p);
  return ret;
}

static ERL_NIF_TERM tensor_to_matrix(ErlNifEnv *env, int argc,
                                     const ERL_NIF_TERM argv[]) {
  unsigned i, j;
  ERL_NIF_TERM ret;
  TF_Tensor **tensor;
  mx_t mx;
  mx.p = NULL;
  enif_get_resource(env, argv[0], tensor_resource, (void *)&tensor);
  TF_DataType type = TF_TensorType(*tensor);
  if (TF_NumDims(*tensor) == 2) {
    mx.p = alloc_matrix(env,
                        (unsigned)(TF_Dim(*tensor, (TF_NumDims(*tensor) - 2))),
                        (unsigned)(TF_Dim(*tensor, (TF_NumDims(*tensor) - 1))));

    if (type == TF_FLOAT) {
      float *float_tensor_data = (float *)TF_TensorData(*tensor);
      for (j = 0; j < (unsigned)(TF_Dim(*tensor, (TF_NumDims(*tensor) - 2)));
           j++) {
        for (i = 0; i < (unsigned)(TF_Dim(*tensor, (TF_NumDims(*tensor) - 1)));
             i++) {
          POS(mx.p, j, i) = (double)*float_tensor_data++;
        }
      }
    }

    else if (type == TF_INT32) {
      int32_t *int32_tensor_data = (int32_t *)TF_TensorData(*tensor);
      for (j = 0; j < (unsigned)(TF_Dim(*tensor, (TF_NumDims(*tensor) - 2)));
           j++) {
        for (i = 0; i < (unsigned)(TF_Dim(*tensor, (TF_NumDims(*tensor) - 1)));
             i++) {
          POS(mx.p, j, i) = (double)*int32_tensor_data++;
        }
      }
    }

    else if (type == TF_DOUBLE) {
      double *double_tensor_data = (double *)TF_TensorData(*tensor);
      for (j = 0; j < (unsigned)(TF_Dim(*tensor, (TF_NumDims(*tensor) - 2)));
           j++) {
        for (i = 0; i < (unsigned)(TF_Dim(*tensor, (TF_NumDims(*tensor) - 1)));
             i++) {
          POS(mx.p, j, i) = *double_tensor_data++;
        }
      }
    }
  } else
    return enif_make_badarg(env);

  ret = enif_make_resource(env, mx.p);
  enif_release_resource(mx.p);
  return ret;
}

static ErlNifFunc nif_funcs[] = {
    {"create_matrix", 3, create_matrix},
    {"matrix_pos", 3, matrix_pos},
    {"append_to_matrix", 2, append_to_matrix},
    {"size_of_matrix", 1, size_of_matrix},
    {"matrix_to_lists", 1, matrix_to_lists},
    {"version", 0, version},
    {"read_graph", 1, read_graph},
    {"get_graph_ops", 1, get_graph_ops},
    {"float64_tensor", 2, float64_tensor},
    {"float64_tensor", 1, float64_tensor},
    {"float32_tensor", 2, float32_tensor},
    {"float32_tensor", 1, float32_tensor},
    {"int32_tensor", 2, int32_tensor},
    {"int32_tensor", 1, int32_tensor},
    {"string_tensor", 1, string_tensor},
    {"tensor_datatype", 1, tensor_datatype},
    {"float64_tensor_alloc", 1, float64_tensor_alloc},
    {"float32_tensor_alloc", 1, float32_tensor_alloc},
    {"int32_tensor_alloc", 1, int32_tensor_alloc},
    {"run_session", 5, run_session},
    {"load_image_as_tensor", 1, load_image_as_tensor},
    {"load_csv_as_matrix", 3, load_csv_as_matrix},
    {"add_scalar_to_matrix", 2, add_scalar_to_matrix},
    {"subtract_scalar_from_matrix", 2, subtract_scalar_from_matrix},
    {"multiply_matrix_with_scalar", 2, multiply_matrix_with_scalar},
    {"divide_matrix_by_scalar", 2, divide_matrix_by_scalar},
    {"add_matrices", 2, add_matrices},
    {"subtract_matrices", 2, subtract_matrices},
    {"tensor_to_matrix", 1, tensor_to_matrix},
};

ERL_NIF_INIT(Elixir.Tensorflex.NIFs, nif_funcs, res_loader, NULL, NULL, NULL)
