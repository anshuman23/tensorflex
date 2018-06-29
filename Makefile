CFLAGS = -g -O3 -Wall
ERLANG_PATH = $(shell erl -eval 'io:format("~s", [lists:concat([code:root_dir(), "/erts-", erlang:system_info(version), "/include"])])' -s init stop -noshell)
LIBTENSORFLOW_PATH = /usr/local/lib
CFLAGS += -I$(ERLANG_PATH)
CFLAGS += -Ic_src
LDFLAGS += -L$(LIBTENSORFLOW_PATH)
ifeq ($(shell uname -s), Darwin)
	LDFLAGS += -flat_namespace -undefined suppress
endif
LIB_SO_NAME = priv/Tensorflex.so
CFLAGS += -fPIC
NIF=c_src/Tensorflex.c

$(LIB_SO_NAME): $(NIF)
	mkdir -p priv
	$(CC) $(CFLAGS) -shared $(LDFLAGS) $^ -ltensorflow -o $@
