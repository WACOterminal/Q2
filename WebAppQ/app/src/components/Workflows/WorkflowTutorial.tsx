// WebAppQ/app/src/components/Workflows/WorkflowTutorial.tsx
import React from 'react';
import Joyride, { Step } from 'react-joyride';

interface WorkflowTutorialProps {
  run: boolean;
  callback: (data: any) => void;
}

export const WorkflowTutorial: React.FC<WorkflowTutorialProps> = ({ run, callback }) => {
  const steps: Step[] = [
    {
      target: '.dndnode:nth-child(2)',
      content: 'This is the sidebar. You can drag nodes from here onto the canvas to build your workflow.',
      placement: 'right',
    },
    {
      target: '.reactflow-wrapper',
      content: 'This is the canvas. Drop nodes here and connect them to create your workflow logic.',
      placement: 'center',
    },
    {
        target: '.dndnode:nth-child(2)',
        content: 'Let\'s start by dragging a "Task Node" onto the canvas.',
        placement: 'right',
    },
    {
      target: '.react-flow__node-default',
      content: 'Great! Now, let\'s select our new node to configure its properties in the details panel.',
      placement: 'right',
    },
    {
      target: '.node-detail-panel',
      content: 'Here you can define the specifics of the task, such as the prompt and which agent should execute it.',
      placement: 'left',
    },
    {
      target: 'button:contains("Save Workflow")',
      content: 'Once you are happy with your workflow, click here to save it.',
      placement: 'top',
    },
  ];

  return (
    <Joyride
      steps={steps}
      run={run}
      continuous
      showProgress
      showSkipButton
      callback={callback}
      styles={{
        options: {
          primaryColor: '#4CAF50',
        },
      }}
    />
  );
}; 