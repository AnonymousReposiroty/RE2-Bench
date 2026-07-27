from sympy.core.random import _assumptions_shuffle as shuffle

def _ask(fact, obj):
    """
    Find the truth value for a property of an object.

    This function is called when a request is made to see what a fact
    value is.

    For this we use several techniques:

    First, the fact-evaluation function is tried, if it exists (for
    example _eval_is_integer). Then we try related facts. For example

        rational   -->   integer

    another example is joined rule:

        integer & !odd  --> even

    so in the latter case if we are looking at what 'even' value is,
    'integer' and 'odd' facts will be asked.

    In all cases, when we settle on some fact value, its implications are
    deduced, and the result is cached in ._assumptions.
    """
    # FactKB which is dict-like and maps facts to their known values:
    assumptions = obj._assumptions

    # A dict that maps facts to their handlers:
    handler_map = obj._prop_handler

    # This is our queue of facts to check:
    facts_to_check = [fact]
    facts_queued = {fact}

    # Loop over the queue as it extends
    for fact_i in facts_to_check:

        # If fact_i has already been determined then we don't need to rerun the
        # handler. There is a potential race condition for multithreaded code
        # though because it's possible that fact_i was checked in another
        # thread. The main logic of the loop below would potentially skip
        # checking assumptions[fact] in this case so we check it once after the
        # loop to be sure.
        if fact_i in assumptions:
            continue

        # Now we call the associated handler for fact_i if it exists.
        fact_i_value = None
        handler_i = handler_map.get(fact_i)
        if handler_i is not None:
            fact_i_value = handler_i(obj)

        # If we get a new value for fact_i then we should update our knowledge
        # of fact_i as well as any related facts that can be inferred using the
        # inference rules connecting the fact_i and any other fact values that
        # are already known.
        if fact_i_value is not None:
            assumptions.deduce_all_facts(((fact_i, fact_i_value),))

        # Usually if assumptions[fact] is now not None then that is because of
        # the call to deduce_all_facts above. The handler for fact_i returned
        # True or False and knowing fact_i (which is equal to fact in the first
        # iteration) implies knowing a value for fact. It is also possible
        # though that independent code e.g. called indirectly by the handler or
        # called in another thread in a multithreaded context might have
        # resulted in assumptions[fact] being set. Either way we return it.
        fact_value = assumptions.get(fact)
        if fact_value is not None:
            return fact_value

        # Extend the queue with other facts that might determine fact_i. Here
        # we randomise the order of the facts that are checked. This should not
        # lead to any non-determinism if all handlers are logically consistent
        # with the inference rules for the facts. Non-deterministic assumptions
        # queries can result from bugs in the handlers that are exposed by this
        # call to shuffle. These are pushed to the back of the queue meaning
        # that the inference graph is traversed in breadth-first order.
        new_facts_to_check = list(_assume_rules.prereq[fact_i] - facts_queued)
        shuffle(new_facts_to_check)
        facts_to_check.extend(new_facts_to_check)
        facts_queued.update(new_facts_to_check)

    # The above loop should be able to handle everything fine in a
    # single-threaded context but in multithreaded code it is possible that
    # this thread skipped computing a particular fact that was computed in
    # another thread (due to the continue). In that case it is possible that
    # fact was inferred and is now stored in the assumptions dict but it wasn't
    # checked for in the body of the loop. This is an obscure case but to make
    # sure we catch it we check once here at the end of the loop.
    if fact in assumptions:
        return assumptions[fact]

    # This query can not be answered. It's possible that e.g. another thread
    # has already stored None for fact but assumptions._tell does not mind if
    # we call _tell twice setting the same value. If this raises
    # InconsistentAssumptions then it probably means that another thread
    # attempted to compute this and got a value of True or False rather than
    # None. In that case there must be a bug in at least one of the handlers.
    # If the handlers are all deterministic and are consistent with the
    # inference rules then the same value should be computed for fact in all
    # threads.
    assumptions._tell(fact, None)
    return None
