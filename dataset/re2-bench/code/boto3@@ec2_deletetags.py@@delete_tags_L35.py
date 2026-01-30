def delete_tags(self, **kwargs):
    kwargs['Resources'] = [self.id]
    return self.meta.client.delete_tags(**kwargs)
