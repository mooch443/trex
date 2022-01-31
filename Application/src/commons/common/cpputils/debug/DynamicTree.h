#ifndef _DYNAMIC_TREE_H
#define _DYNAMIC_TREE_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>


template <class T>
class TreeNode {
public:
    //! node value
    T value; 

    //! node value for ordering
    size_t offset;

    /**
     * The anchor for this nodes children. 
     * NULL if no children present.
     **/
    TreeNode<T> *first_child, *last_child, *parent;

    /**
     * Next and prev are the next and previous elements in the layer of this node.
     **/
    TreeNode<T> *next, *prev;

    /**
     * Constructor.
     * @param v the value of the node
     * @param o ordering value
     **/
    TreeNode(T v, size_t o) : value(v), offset(o),
    	first_child(NULL), last_child(NULL), parent(NULL),
    	next(NULL), prev(NULL)
    { }

    /**
     * Destructor.
     * Removes everything thats below this node.
     **/
    ~TreeNode() {
        deleteChildren();
        removeFromList();
    }
    
protected:
	/**
	 * Remove all traces of this node from the list of its parent.
	 */
	void removeFromList() {
		// remove this element from list
        if (next) {
            next->prev = prev;
        }
        if(prev) {
        	prev->next = next;
        }
        
        if(parent && parent->first_child == this) {
        	parent->first_child = next;
        }
        if(parent && parent->last_child == this) {
        	parent->last_child = prev;
        }
        
        parent = NULL;
        next = NULL;
        prev = NULL;
    }
    
public:
    /**
     * Delete all children of a node.
     */
    void deleteChildren() {
    	// recursively remove all children
        TreeNode<T> *e;
        while ((e = first_child))
            delete e;
    }
    
    bool is_child_of(TreeNode<T>* e) {
        auto n = parent;
        while(n) {
            if(n == e)
                return true;
            n = n->parent;
        }
        return false;
    }

    /**
     * Calculate biggest offset.
     **/
    size_t maxOffset() {
        // use rightmost child, because it has to have the biggest offset
        // then check its children for bigger offsets - if there a no children,
        // return this nodes offset
        return last_child ? last_child->maxOffset() : offset;
    }

    /**
     * Find the layer where the given offset is contained.
     * Given this tree (node number and its offset in brackets):
     *
     *                       root
     *						/    \
     *					  1(2)   2(20)
     *						|
     *					  3(10)
     *
     * For an offset value of 10 the node 3 will be returned.
     * For an offset value of 2 the node 1 will be returned.
     * For an offset value of 5 the node 1 will be returned.
     * For an offset value of 15 the root node will be returned.
     *
     * @param o the offset to be tested
     * @return NULL if no values were found
     **/
    TreeNode<T> * nodeClosestTo(size_t o) {
        // if the offset of this node is already bigger than the searched offset, return parent (if any) or NULL
        if (this->offset > o)  return this->parent; 
        if (this->offset == o) return this;

        // is there already something in this tree?
        if (!first_child || last_child->maxOffset() < o) {
            //printf("-- no children at %d or already bigger than maxOffset of last child\n", this->offset);
            return this;
        }
        else {
            //printf("\nNow recursing at %d for %d\n", this->offset, o);
            TreeNode<T> *found_node = this;

            // have to iterate through all elements to get the correct
            // position for inserting (breadth-first search):
            TreeNode<T> *e = first_child, *tmp = NULL;
            while (e || tmp) {
                // if this is true we either have found an element with a bigger offset, which forces us
                // to look at the previous element, or we reached the end of the list and have to return the
                // last element:
                if ((e && e->offset > o) || (!e && tmp)) {
                    if (tmp) {
                        // the value is not inside the previous node, just append to this one
                        if (tmp->maxOffset() < o) {
                            found_node = this;
                            //printf("-- found_node = this\n");
                        }
                        else {
                            // e's offset is bigger, tmps offset is smaller - the maximum offset in tmp is bigger than the searched offset
                            // so we have to recurse down:
                            // the correct value is somewhere inside the last nodes' subtree
                            found_node = tmp->nodeClosestTo(o);
                            //printf("-- found_node = tmp->nodeClosestTo(%d) [tmp = %d]\n", o, tmp->offset);
                        }
                    }

                    break;
                }

                tmp = e;
                if (e) e = e->next;
            }

            return found_node;
        }
    }

    /**
     * Insert element into tree structure as a child of the current Element.
     * Will insert the element according to its order value (if possible as direct
     * child. if the offset value requires the node to be inserted into a lower node,
     * it will be done).
     * @param insert the node to be inserted
     **/
    TreeNode<T> * addChild(TreeNode<T> *insert) {
        assert(insert);
        if(insert->offset < offset)
            return NULL;
        
        // remove node from a possible previous list and
        // reset its handles:
        insert->removeFromList();
        
        // parent that will be returned - by default it is this object
        TreeNode<T> *found_parent = this;

        // is there already something in this tree?
        if (!first_child) {
            first_child = last_child = insert;
            insert->prev = NULL;
            insert->next = NULL;
        }
        else {
            // append to to list - is the last childs value smaller than the current one?
            // (depends on ordered list)
            if (last_child && last_child->maxOffset() <= insert->offset) {
                last_child->next = insert;
                insert->prev = last_child;
                last_child = insert;
            }
            else {
                // have to iterate through all elements to get the correct
                // position for inserting (breadth-first search):
                TreeNode<T> *e = first_child, *tmp = NULL;
                while (e || tmp) {
                    // if this is true, the last node (or its children) had the optimal offset value for
                    // insertion
                    if ((e && e->offset > insert->offset) || (!e && tmp)) {
                        if (tmp) {
                            // insert between e and its predecessor if maximum offset of tmp is <= insert->offset
                            if (tmp->maxOffset() <= insert->offset) {
                                insert->prev = tmp;
                                insert->next = tmp->next;
                                tmp->next = insert;
                            }
                            else {
                                // now we have to recursively search for the right place to fit this value in...
                                found_parent = tmp->addChild(insert);
                            }
                        }
                        else {
                            // prepend to list because it is smaller than everything in this list (== smaller than first node)
                            // (assuming sorted list)
                            insert->next = first_child;
                            insert->prev = NULL;
                            first_child->prev = insert;
                            first_child = insert;
                        }
                        
                        break;
                    }

                    tmp = e;
                    if(e) e = e->next;
                }
            }
        }

        insert->parent = found_parent;
        return found_parent;
    }

    void print(const char * tabs = "") {
        char * t = new char[strlen(tabs) + 1];
        strcpy(t, tabs);
        strcat(t, "\t");

        auto e = this;
        while (e) {
            printf("%s%ld(%d/'%c')\n", tabs, e->offset, e->value, e->value);
            if (e->first_child) {
                e->first_child->print(t);
            }
            e = e->next;
        }
    }
};

template <class T> 
class OrderedTree {
public:
    /**
     * Constructor.
     **/
    OrderedTree() : anchor(NULL), last(NULL) {
        
    }

    /**
     * Constructor.
     * Will use the given node as root node.
     * @param root node to be used as root
     **/
    OrderedTree(TreeNode<T> *root) : anchor(root), last(root) {
    }

    /**
     * Destructor. Releases all child elements.
     **/
    ~OrderedTree();

    //! Adds node to the tree, sorted by its offset value, creates the node with given values
    TreeNode<T>* addNode(T v, size_t offset);
    
    //! Adds node to the tree, sorted by its offset value
    bool addNode(TreeNode<T> *node);

    /**
     * Returns the node for a given offset. If the absolute offset
     * isnt part of this tree, then the closest value will be returned.
     * This may be NULL for no entries, or the closest value smaller than
     * the given value.
     **/
    TreeNode<T>* nodeForOffset(size_t offset);

    //! returns root node
    TreeNode<T>* root() { return anchor; }

private:
    //! root node
    TreeNode<T> *anchor;

    //! last created node (for performance, e.g.: go up one level, add new node, etc.)
    TreeNode<T> *last;
};


/*

    I M P L E M E N T A T I O N

*/

template <class T>
OrderedTree<T>::~OrderedTree() {
    if (anchor) delete anchor; // will recursively remove all elements
}

template <class T>
bool OrderedTree<T>::addNode(TreeNode<T> *node) {
    // if tree has got a node, insert there
    if (!root()) {
        anchor = node;
        last = anchor;
    }
    else {
        if(!root()->addChild(node))
            return false;
    }
    
    return true;
}

template <class T>
TreeNode<T>* OrderedTree<T>::addNode(T v, size_t offset) {
    TreeNode<T> *node = new TreeNode<T>(v, offset);
    addNode(node);

    return node;
}

template <class T>
TreeNode<T>* OrderedTree<T>::nodeForOffset(size_t offset) {
    if(root()) return root()->nodeClosestTo(offset);
    return NULL;
}


#endif
