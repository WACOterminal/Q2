import React, { useEffect, useState, useContext } from 'react';
import { BrowserRouter as Router, Routes, Route, Link as RouterLink, useNavigate } from 'react-router-dom';
import { AppBar, Toolbar, Typography, Button, Box, IconButton, Drawer, List, ListItem, ListItemButton, ListItemIcon, ListItemText, Divider, CssBaseline } from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import HomeIcon from '@mui/icons-material/Home';
import ChatIcon from '@mui/icons-material/Chat';
import AccountTreeIcon from '@mui/icons-material/AccountTree';
import InsightsIcon from '@mui/icons-material/Insights';
import GraphIcon from '@mui/icons-material/DataObject'; // Using DataObject for Knowledge Graph
import CodeIcon from '@mui/icons-material/Code'; // Using Code for Agent Sandbox
import PolicyIcon from '@mui/icons-material/Policy'; // For Governance
import QueryStatsIcon from '@mui/icons-material/QueryStats'; // For MLOps Dashboard
import PeopleIcon from '@mui/icons-material/People'; // For Human-Agent Teaming
import AuthContext from './AuthContext';
import Chat from './pages/Chat';
import Home from './pages/Home';
import WorkflowVisualizer from './pages/WorkflowVisualizer';
import KnowledgeGraph from './pages/KnowledgeGraph';
import AgentSandbox from './pages/AgentSandbox';
import Governance from './pages/Governance';
import MLDashboard from './pages/MLDashboard'; // Import the new MLDashboard component
import HumanAgentTeaming from './pages/HumanAgentTeaming'; // Import the new Human-Agent Teaming component

const drawerWidth = 240;

interface NavItem {
  text: string;
  icon: React.ReactElement;
  path: string;
}

const navItems: NavItem[] = [
  { text: 'Home', icon: <HomeIcon />, path: '/' },
  { text: 'Chat', icon: <ChatIcon />, path: '/chat' },
  { text: 'Workflows', icon: <AccountTreeIcon />, path: '/workflows' },
  { text: 'Knowledge Graph', icon: <GraphIcon />, path: '/knowledge-graph' },
  { text: 'Agent Sandbox', icon: <CodeIcon />, path: '/agent-sandbox' },
  { text: 'Governance', icon: <PolicyIcon />, path: '/governance' },
  { text: 'MLOps Dashboard', icon: <QueryStatsIcon />, path: '/mlops-dashboard' }, // Add MLOps Dashboard link
  { text: 'Human-Agent Teaming', icon: <PeopleIcon />, path: '/human-agent-teaming' }, // Add Human-Agent Teaming link
];

const App: React.FC = () => {
  const { keycloak, authenticated } = useContext(AuthContext);
  const navigate = useNavigate();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [isAuthChecked, setIsAuthChecked] = useState(false); // New state to track if auth check is done

  useEffect(() => {
    // This effect ensures navigation happens only once after authentication state is confirmed
    if (keycloak && typeof authenticated === 'boolean') {
      setIsAuthChecked(true);
      if (authenticated) {
        // Optionally navigate to a default authenticated route or just stay
        console.log('User is authenticated. Token:', keycloak.tokenParsed);
      } else {
        // If not authenticated, ensure we are at the base path or handle login
        console.log('User is not authenticated.');
        // keycloak.login(); // Uncomment to force login if needed
      }
    }
  }, [keycloak, authenticated, navigate]);

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const handleLogout = () => {
    keycloak?.logout({
      redirectUri: window.location.origin, // Redirects to the root of your app after logout
    });
  };

  const drawer = (
    <Box onClick={handleDrawerToggle} sx={{ textAlign: 'center' }}>
      <Typography variant="h6" sx={{ my: 2 }}>
        Q Platform
      </Typography>
      <Divider />
      <List>
        {navItems.map((item) => (
          <ListItem key={item.text} disablePadding>
            <ListItemButton component={RouterLink} to={item.path} sx={{ textAlign: 'left' }}>
              <ListItemIcon>{item.icon}</ListItemIcon>
              <ListItemText primary={item.text} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
    </Box>
  );

  // Only render content after authentication status is checked
  if (!isAuthChecked) {
    return <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>Loading authentication...</Box>;
  }

  return (
    <Box sx={{ display: 'flex' }}>
      <CssBaseline />
      <AppBar component="nav" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { sm: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
          <Typography
            variant="h6"
            component="div"
            sx={{ flexGrow: 1, display: { xs: 'none', sm: 'block' } }}
          >
            Q Platform
          </Typography>
          <Box sx={{ display: { xs: 'none', sm: 'block' } }}>
            {authenticated ? (
              <Button color="inherit" onClick={handleLogout}>
                Logout
              </Button>
            ) : (
              <Button color="inherit" onClick={() => keycloak?.login()}>
                Login
              </Button>
            )}
          </Box>
        </Toolbar>
      </AppBar>
      <Box component="nav">
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{
            keepMounted: true, // Better open performance on mobile.
          }}
          sx={{
            display: { xs: 'block', sm: 'none' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
          }}
        >
          {drawer}
        </Drawer>
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: 'none', sm: 'block' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
          }}
          open
        >
          <Toolbar /> {/* Spacer for AppBar */}
          {drawer}
        </Drawer>
      </Box>
      <Box component="main" sx={{ flexGrow: 1, p: 3, width: { sm: `calc(100% - ${drawerWidth}px)` } }}>
        <Toolbar /> {/* Spacer for AppBar */}
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/chat" element={authenticated ? <Chat /> : <Typography>Please login to access the chat.</Typography>} />
          <Route path="/workflows" element={authenticated ? <WorkflowVisualizer /> : <Typography>Please login to access workflows.</Typography>} />
          <Route path="/knowledge-graph" element={authenticated ? <KnowledgeGraph /> : <Typography>Please login to access the knowledge graph.</Typography>} />
          <Route path="/agent-sandbox" element={authenticated ? <AgentSandbox /> : <Typography>Please login to access the agent sandbox.</Typography>} />
          <Route path="/governance" element={authenticated ? <Governance /> : <Typography>Please login to access governance.</Typography>} />
          <Route path="/mlops-dashboard" element={authenticated ? <MLDashboard /> : <Typography>Please login to access the MLOps Dashboard.</Typography>} /> {/* Add MLDashboard route */}
          <Route path="/human-agent-teaming" element={authenticated ? <HumanAgentTeaming /> : <Typography>Please login to access Human-Agent Teaming.</Typography>} /> {/* Add Human-Agent Teaming route */}
          {/* Add other routes here */}
        </Routes>
      </Box>
    </Box>
  );
};

export default App;
