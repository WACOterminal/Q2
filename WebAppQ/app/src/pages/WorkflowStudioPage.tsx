import React, { useState, useEffect } from 'react';
import { Box, Button, Typography, Paper } from '@mui/material';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import { WorkflowBuilder } from '../components/Workflows/WorkflowBuilder';
import { GenerateWorkflowModal } from '../components/Workflows/GenerateWorkflowModal';

const sampleYaml = `
workflow_id: "wf_example"
original_prompt: "An example workflow to demonstrate the studio."
shared_context:
  service_name: "example-service"

tasks:
  - task_id: "example_task_1"
    type: "task"
    agent_personality: "default"
    prompt: "This is the first task."
    dependencies: []
`.trim();

export function WorkflowStudioPage() {
    const [yamlContent, setYamlContent] = useState(sampleYaml);
    const [isModalOpen, setIsModalOpen] = useState(false);

    const handleSave = (newYaml: string) => {
        // In a real application, this would save the workflow to the backend
        console.log("Saving workflow:", newYaml);
        setYamlContent(newYaml);
    };

    const handleWorkflowGenerated = (generatedYaml: string) => {
        setYamlContent(generatedYaml);
        setIsModalOpen(false); // Close modal after generation
    };

    return (
        <Box sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h4">
                    Workflow Studio
                </Typography>
                <Button
                    variant="contained"
                    startIcon={<AutoAwesomeIcon />}
                    onClick={() => setIsModalOpen(true)}
                >
                    Create with AI
                </Button>
            </Box>
            
            <WorkflowBuilder initialYaml={yamlContent} onSave={handleSave} />

            <GenerateWorkflowModal
                open={isModalOpen}
                onClose={() => setIsModalOpen(false)}
                onWorkflowGenerated={handleWorkflowGenerated}
            />
        </Box>
    );
} 