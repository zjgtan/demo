BUILDDIR=./build


all: $(BUILDDIR) main test_base

$(BUILDDIR):
	mkdir -p $(BUILDDIR)/bin $(BUILDDIR)/lib $(BUILDDIR)/obj

main: $(BUILDDIR)/obj/main.o $(BUILDDIR)/obj/base.o
	g++ $^ -o $(BUILDDIR)/bin/$@

test_base: $(BUILDDIR)/obj/test_base.o $(BUILDDIR)/obj/base.o
	g++ $^ -o $(BUILDDIR)/bin/test_base

$(BUILDDIR)/obj/%.o: ./src/%.cpp
	g++ -c $< -o $@ -I./include

$(BUILDDIR)/obj/%.o: ./test/%.cpp
	g++ -c $< -o $@ -I./include

