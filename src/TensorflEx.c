#include "erl_nif.h"
#include "c_api.h"

static ERL_NIF_TERM version(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  return enif_make_string(env, TF_Version() , ERL_NIF_LATIN1);
}

static ErlNifFunc nif_funcs[] =
  {
    { "version", 0, version }
  };

ERL_NIF_INIT(Elixir.TensorflEx, nif_funcs, NULL, NULL, NULL, NULL)

