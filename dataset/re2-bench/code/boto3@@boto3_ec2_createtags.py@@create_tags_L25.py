def create_tags(self, **kwargs):
    # Call the client method
    self.meta.client.create_tags(**kwargs)
    resources = kwargs.get('Resources', [])
    tags = kwargs.get('Tags', [])
    tag_resources = []

    # Generate all of the tag resources that just were created with the
    # preceding client call.
    for resource in resources:
        for tag in tags:
            # Add each tag from the tag set for each resource to the list
            # that is returned by the method.
            tag_resource = self.Tag(resource, tag['Key'], tag['Value'])
            tag_resources.append(tag_resource)
    return tag_resources
