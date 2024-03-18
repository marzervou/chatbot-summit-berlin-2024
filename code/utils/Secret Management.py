# Databricks notebook source
# Only for instructors to run!!

import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace

w = WorkspaceClient()
scope_name = "llm-scope"
w.secrets.create_scope(scope=scope_name)

key_name = "chain-key"
w.secrets.put_secret(scope=scope_name, key=key_name, string_value=config['PAT_TOKEN'])

# Now allow manage access on the scope for all the users of the lab
# user_list = ["odl_user_1237277@databrickslabs.com"]
# for user in user_list:
#   w.secrets.put_acl(scope=scope_name, permission=workspace.AclPermission.MANAGE, principal=user)


