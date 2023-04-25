#define PY_SSIZE_T_CLEAN
#include <Python.h>

static long
impl_adjacent(long p, long size, long adj[4])
{
    long l;

    l = 0;

    if (p % size != 0)
        adj[l++] = p - 1;
    if (p % size != size - 1)
        adj[l++] = p + 1;

    if (p / size != 0)
        adj[l++] = p - size;
    if (p / size != size - 1)
        adj[l++] = p + size;

    return l;
}

static PyObject *
wine_adjacent(PyObject *self, PyObject *args)
{
    long p, size;
    long i, l, adj[4];
    PyObject *item, *result;

    if (!PyArg_ParseTuple(args, "ll", &p, &size))
        return NULL;

    l = impl_adjacent(p, size, adj);

    result = PyList_New(l);
    for (i = 0; i < l; ++i) {
        item = PyLong_FromLong(adj[i]);
        PyList_SET_ITEM(result, i, item);
    }

    return result;
}

static PyMethodDef WineMethods[] = {
    {"adjacent", wine_adjacent, METH_VARARGS,
     "vine.adjacent"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef wine = {
    PyModuleDef_HEAD_INIT,
    "wine",
    NULL,
    -1,
    WineMethods
};

PyMODINIT_FUNC
PyInit_wine(void)
{
    return PyModule_Create(&wine);
}
