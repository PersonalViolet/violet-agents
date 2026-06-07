from typing import Dict, Any, List, Optional, Union

try:
    from fastmcp import Client, FastMCP
    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False



class MCPClient(Client):

    def a():
        print("a")

        

