#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "arrayobject.h"

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

static void
impl_group(long p, long size, long c, long *data, long *gl, long *sl,
           long *queue, long *visit, long *group, long *space)
{
    long i, j, f, b, q, l, m, n, adj[4];

    i = 0, j = 0, f = 0, b = 0;

    queue[0] = p;
    visit[p] = 1;

    while (f <= b) {
        q = queue[f++];
        group[i++] = q;

        l = impl_adjacent(q, size, adj);
        for (m = 0; m < l; ++m) {
            n = adj[m];

            if (visit[n])
                continue;

            visit[n] = 1;

            if (data[n] == 0)
                space[j++] = n;
            if (data[n] == c)
                queue[++b] = n;
        }
    }

    *gl = i;
    *sl = j;
}

static PyObject *
wine_group(PyObject *self, PyObject *args)
{
    long p, size, c;
    PyObject *obj, *raw;
    PyObject *item, *rgroup, *rspace, *result;

    if (!PyArg_ParseTuple(args, "lllO!", &p, &size, &c, &PyArray_Type, &obj))
        return NULL;

    if ((raw = PyArray_FROM_OTF(obj, NPY_LONG, NPY_ARRAY_IN_ARRAY)) == NULL)
        return NULL;

    npy_intp dims;
    long *data, *queue, *visit, *group, *space;
    long gl, sl, i;

    dims = PyArray_DIMS((PyArrayObject*)raw)[0];
    data = (long *)PyArray_DATA((PyArrayObject*)raw);

    queue = (long *)calloc(dims, sizeof(long));
    visit = (long *)calloc(dims, sizeof(long));
    group = (long *)calloc(dims, sizeof(long));
    space = (long *)calloc(dims, sizeof(long));

    impl_group(p, size, c, data, &gl, &sl, queue, visit, group, space);

    rgroup = PyList_New(gl);
    for (i = 0; i < gl; ++i) {
        item = PyLong_FromLong(group[i]);
        PyList_SET_ITEM(rgroup, i, item);
    }

    rspace = PyList_New(sl);
    for (i = 0; i < sl; ++i) {
        item = PyLong_FromLong(space[i]);
        PyList_SET_ITEM(rspace, i, item);
    }

    result = PyTuple_New(2);
    PyTuple_SET_ITEM(result, 0, rgroup);
    PyTuple_SET_ITEM(result, 1, rspace);

    free(queue);
    free(visit);
    free(group);
    free(space);

    return result;
}

static long
impl_capture(long p, long size, long c, long *data, long *output,
             long *queue, long *visit, long *group, long *space)
{
    long cl, gl, sl, i, l, m, n, adj[4];

    cl = 0;

    l = impl_adjacent(p, size, adj);
    for (m = 0; m < l; ++m) {
        n = adj[m];

        if (data[n] != c)
            continue;

        if (visit[n])
            continue;

        impl_group(n, size, c, data, &gl, &sl, queue, visit, group, space);

        if (!sl) {
            for (i = 0; i < gl; ++i)
                output[cl++] = group[i];
        }
    }

    return cl;
}

static PyObject *
wine_capture(PyObject *self, PyObject *args)
{
    long p, size, c;
    PyObject *obj, *raw;
    PyObject *item, *result;

    if (!PyArg_ParseTuple(args, "lllO!", &p, &size, &c, &PyArray_Type, &obj))
        return NULL;

    if ((raw = PyArray_FROM_OTF(obj, NPY_LONG, NPY_ARRAY_IN_ARRAY)) == NULL)
        return NULL;

    npy_intp dims;
    long *data, *queue, *visit, *group, *space, *output;
    long i, l;

    dims = PyArray_DIMS((PyArrayObject*)raw)[0];
    data = (long *)PyArray_DATA((PyArrayObject*)raw);

    queue = (long *)calloc(dims, sizeof(long));
    visit = (long *)calloc(dims, sizeof(long));
    group = (long *)calloc(dims, sizeof(long));
    space = (long *)calloc(dims, sizeof(long));
    output = (long *)calloc(dims, sizeof(long));

    l = impl_capture(p, size, c, data, output, queue, visit, group, space);

    result = PyList_New(l);
    for (i = 0; i < l; ++i) {
        item = PyLong_FromLong(output[i]);
        PyList_SET_ITEM(result, i, item);
    }

    free(queue);
    free(visit);
    free(group);
    free(space);
    free(output);

    return result;
}

static long
impl_illegal(long p, long size, long c, long *data,
             long *queue, long *visit, long *space)
{
    long j, f, b, q, l, m, n, adj[4];

    l = impl_adjacent(p, size, adj);
    for (m = 0; m < l; ++m)
        if (data[adj[m]] == 0)
            return 0;

    j = 0, f = 0, b = 0;

    queue[0] = p;
    visit[p] = 1;

    while (f <= b) {
        q = queue[f++];

        l = impl_adjacent(q, size, adj);
        for (m = 0; m < l; ++m) {
            n = adj[m];

            if (visit[n])
                continue;

            visit[n] = 1;

            if (data[n] == 0)
                space[j++] = n;
            if (data[n] == c)
                queue[++b] = n;
        }
    }

    return j == 0;
}

static PyObject *
wine_illegal(PyObject *self, PyObject *args)
{
    long p, size, c, r;
    PyObject *obj, *raw;

    if (!PyArg_ParseTuple(args, "lllO!", &p, &size, &c, &PyArray_Type, &obj))
        return NULL;

    if ((raw = PyArray_FROM_OTF(obj, NPY_LONG, NPY_ARRAY_IN_ARRAY)) == NULL)
        return NULL;

    npy_intp dims;
    long *data, *queue, *visit, *space;

    dims = PyArray_DIMS((PyArrayObject*)raw)[0];
    data = (long *)PyArray_DATA((PyArrayObject*)raw);

    queue = (long *)calloc(dims, sizeof(long));
    visit = (long *)calloc(dims, sizeof(long));
    space = (long *)calloc(dims, sizeof(long));

    r = impl_illegal(p, size, c, data, queue, visit, space);

    free(queue);
    free(visit);
    free(space);

    if (r) Py_RETURN_TRUE;
    else Py_RETURN_FALSE;
}

static PyMethodDef WineMethods[] = {
    {"adjacent", wine_adjacent, METH_VARARGS,
     "vine.adjacent"},
    {"group", wine_group, METH_VARARGS,
     "vine.group"},
    {"capture", wine_capture, METH_VARARGS,
     ""},
    {"illegal", wine_illegal, METH_VARARGS,
     ""},
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
    import_array();

    return PyModule_Create(&wine);
}
