def single_fact_lookup(known_facts_keys, known_facts_cnf):
    # Return the dictionary for quick lookup of single fact
    mapping = {}
    for key in known_facts_keys:
        mapping[key] = {key}
        for other_key in known_facts_keys:
            if other_key != key:
                if ask_full_inference(other_key, key, known_facts_cnf):
                    mapping[key].add(other_key)
                if ask_full_inference(~other_key, key, known_facts_cnf):
                    mapping[key].add(~other_key)
    return mapping
