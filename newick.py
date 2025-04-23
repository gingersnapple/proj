
import time

text = open("/var/home/luka/proj/Papilonidae_dataset_v2/papilionidae_tree_new.txt").read()
N = len(text)

def quote(name, escaped_chars=" \t\r\n()[]':;,"):
    """Return the name quoted if it has any characters that need escaping."""
    if any(c in name for c in escaped_chars):
        return "'%s'" % name.replace("'", "''")  # ' escapes to '' in newicks
    else:
        return name

def unquote(name):
    """Return the name unquoted if it was quoted."""
    name = str(name).strip()

    if name.startswith("'") and name.endswith("'"):  # standard quoting with '
        return name[1:-1].replace("''", "'")  # ' escapes to '' in newicks
    elif name.startswith('"') and name.endswith('"'):  # non-standard quoting "
        return name[1:-1].replace('""', '"')
    else:
        return name

endings=':,);'

def skip_content(pos:int):
    """Return the position where the content ends."""

    while pos < N and text[pos] not in endings:
        pos += 1
    return pos

def read_content(pos:int):
    """Return content starting at position pos in text, and where it ends."""
    # text = '...(node_1:0.5[&&NHX:p=a],...'  ->  'node_1:0.5[&&NHX:p=a]'
    #             ^-- pos              ^-- pos (returned)
    start = pos
    pos = skip_content(pos)
    return text[start:pos], pos


NAME    = {'pname': 'name',    'read': unquote, 'write': quote}
DIST    = {'pname': 'dist',    'read': float,   'write': lambda x: '%g' % float(x)}
SUPPORT = {'pname': 'support', 'read': float,   'write': lambda x: '%g' % float(x)}

parser ={
    'leaf':     [NAME,    DIST],  # ((name:dist)x:y);
    'internal': [SUPPORT, DIST],  # ((x:y)support:dist);
}




class Tree:
    def __init__(self, data=None, children=None):
        """
             :param data: A string or file object with the description of
                 the tree as a newick, or a dict with the contents of a
                 single node.
             :param children: List of nodes to add as children of this one.
             :param parser: A description of how to parse a newick to
                 create a tree. It can be a single number specifying the
                 format or a structure with a fine-grained description of
                 how to interpret nodes (see ``newick.pyx``).

             Examples::

               t1 = Tree()  # empty tree
               t2 = Tree({'name': 'A'})
               t3 = Tree('(A:1,(B:1,(C:1,D:1):0.5):0.5);')
               t4 = Tree(open('/home/user/my-tree.nw'))
             """
        self.up = None
        self.children = children or []

        #self.props = data.copy()

        self.name = data['name']
        self.dist = data['dist']


    @property
    def is_leaf(self):
        """Return True if the current node is a leaf."""
        return not self.children

    # number of nodes in tree rooted in self
    @property
    def branch_size(self):
        i = 0
        for n in traverse(self):
            i += 1
        return i

def read_props(pos:int, is_leaf):
    """Return the properties from the content of a node, and where it ends.

    Example (for the default format of a leaf node):
      'abc:123[&&NHX:x=foo]'  ->  {'name': 'abc', 'dist': 123, 'x': 'foo'}
    """
    #prop0, prop1 = parser['leaf' if is_leaf else 'internal']
    prop0, prop1 = parser['leaf']
    # Shortcuts.
    p0_name, p0_read, p0_req = prop0['pname'], prop0['read'], prop0.get('req')
    p1_name, p1_read, p1_req = prop1['pname'], prop1['read'], prop1.get('req')

    props = {}  # will contain the properties extracted from the content string

    p0_str, pos = read_content(pos)
    #print(p0_str, pos, is_leaf)
    if p0_str:
        props[p0_name] = p0_read(p0_str)

    if pos < N and text[pos] == ':':

        p1_str, pos = read_content(pos+1)
        #print(p1_str, pos)
        props[p1_name] = p1_read(p1_str)

    return props, pos

def read_node(pos:int):
    """Return a node and the position in the text where it ends."""

    if text[pos] == '(':  # node has children
        children, pos = read_nodes(pos)
    else:  # node is a leaf
        children = []

    props, pos = read_props(pos, not children)

    return Tree(props, children), pos


def read_nodes(pos:int):
    """Return a list of nodes and the position in the text where they end."""
    # text looks like '(a,b,c)', where any element can be a list of nodes
    nodes = []
    while pos < N and text[pos] != ')':
        pos += 1  # advance from the separator: "(" or ","

        node, pos = read_node(pos)

        nodes.append(node)

    assert pos < N, 'nodes text ends missing a matching ")"'

    return nodes, pos+1  # it is +1 to advance from the closing ")"


def loads():
    """Return tree from its newick representation."""
    assert text.endswith(';'), 'text ends with no ";"'

    tree, pos = read_node(0)
    # We set check_req=False because the formats requiring certain
    # fields mean it for all nodes but the root.

    assert pos == N - 1, f'root node ends prematurely at {pos}'

    return tree



def traverse(tree):
    """Traverse the tree and yield nodes in pre-order."""
    visiting = [(tree, False)]
    while visiting:
        node, seen = visiting.pop()

        is_leaf = node.is_leaf

        if is_leaf or not seen:
            yield node

        if not seen and not is_leaf:
            visiting.append((node, True))  # add node back, but mark as seen
            visiting += [(n, False) for n in node.children[::-1]]




t0 = time.time()



t = loads()
order = [n for n in traverse(t)]


leaves = [n.name for n in order if n.is_leaf]


print(order[2].branch_size)