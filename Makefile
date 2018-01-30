CFLAGS = -g -O3 -Wall
ERLANG_PATH = /usr/lib/erlang/erts-7.3/include
LIBTENSORFLOW_PATH = /usr/local/lib
CFLAGS += -I$(ERLANG_PATH)
CFLAGS += -Isrc
LDFLAGS += -L$(LIBTENSORFLOW_PATH)
LIB_SO_NAME = priv/TensorflEx.so
CFLAGS += -fPIC
NIF=src/TensorflEx.c

$(LIB_SO_NAME): $(NIF)
	mkdir -p priv
	$(CC) $(CFLAGS) -shared $(LDFLAGS) $^ -ltensorflow -o $@

